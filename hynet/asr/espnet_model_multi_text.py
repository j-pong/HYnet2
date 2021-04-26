from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.lm.abs_model import AbsLM

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        lm: Optional[AbsLM],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None
        
        # Task related
        self.lm = lm
        self.normalize_length = length_normalized_loss
        self.criterion_boot = torch.nn.KLDivLoss(reduction="none")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_aux: torch.Tensor,
        text_aux_lengths: torch.Tensor,
        noisy_label_flag: bool=False,
        inspect: bool=False, 
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            text_aux: (Batch, Length)
            text_aux_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert text_aux_lengths.dim() == 1, text_aux_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == text_aux.shape[0]
            == text_aux_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape, text_aux.shape, text_aux_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
        text_aux = text_aux[:, : text_aux_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # FIXME(dh): Use auxiliary text(text_aux) as you want
        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, decoder_out_att, ys_out_pad_att, acc_att, cer_att, wer_att, repl_ratio, acc_lm_att  = None, None, None, None, None, None, None, None
        else:
            loss_att, decoder_out_att, ys_out_pad_att, acc_att, cer_att, wer_att, repl_ratio, acc_lm_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths, noisy_label_flag
            )

        if inspect:
            decoder_out_att_prob = torch.softmax(decoder_out_att, dim=-1)

            ignore = ys_out_pad_att == self.ignore_id  # (B,)
            total = len(ys_out_pad_att) - ignore.sum().item()
            decoder_out_att = decoder_out_att.masked_fill(ignore.unsqueeze(-1), 0)  # avoid -1 index

            confid_mean = decoder_out_att.sum() / total
            confid_max = decoder_out_att.max()
            confid_min = decoder_out_att.min()
            
        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att 

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
            repl_ratio=repl_ratio,
            acc_lm_att=acc_lm_att,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        text_aux: torch.Tensor=None,
        text_aux_lengths: torch.Tensor=None,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        noisy_label_flag: bool,
    ):
        if not noisy_label_flag:
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        else:
            from espnet.nets.pytorch_backend.nets_utils import pad_list

            _sos = ys_pad.new([self.sos])
            _ignore = ys_pad.new([self.ignore_id])
            
            ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

            ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys]

            ys_in_pad = pad_list(ys_in, self.eos)
            ys_out_pad = pad_list(ys_out, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        
        # 3. Text cleaner for noisy label
        repl_ratio = -1.0
        acc_lm_att = -1.0
        if noisy_label_flag:
            # Lnaguage model probability
            lm_out, _ = self.lm(ys_in_pad, None)

            lm_out_prob = torch.softmax(lm_out, dim=-1)
            decoder_out_prob = torch.softmax(decoder_out, dim=-1)
            total_out_prob = 0.5 * lm_out_prob + decoder_out_prob

            ignore = ys_out_pad.view(-1) == self.ignore_id

            repl_ratio = (decoder_out_prob.argmax(dim=-1) == total_out_prob.argmax(dim=-1)).float().view(-1)
            repl_ratio = repl_ratio.masked_fill(ignore, 0)
            total = len(ys_out_pad.view(-1)) - ignore.sum().item()
            repl_ratio = repl_ratio.sum() / total

            # batch_size = decoder_out.size(0)
            # x = decoder_out.view(-1, self.vocab_size)
            # target = ys_out_pad_lm.view(-1, self.vocab_size)
            
            # with torch.no_grad():
            #     # ignore part is from the noisy target
            #     ignore = ys_out_pad.view(-1, 1) == self.ignore_id  # (B,)
            #     total = len(ys_out_pad) - ignore.sum().item()
            #     x = x.masked_fill(ignore, 0)  # avoid -1 index

            # kl = self.criterion_boot(torch.log_softmax(x, dim=1), target)
            # denom = total if self.normalize_length else batch_size
            # loss_att = loss_att + kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom 
            # loss_att = loss_att / 2        

            acc_lm_att = th_accuracy(
                total_out_prob.view(-1, self.vocab_size),
                ys_out_pad,
                ignore_label=self.ignore_id,
            )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, decoder_out, ys_out_pad, acc_att, cer_att, wer_att, repl_ratio, acc_lm_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError