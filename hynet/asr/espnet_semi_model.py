import logging
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

def calc_confidence(
        probs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Confidence for filtering

        Args:
            probs: (Batch, Length, Vocab_size)
            labels: (Batch, Length)
        """
        bsz, tsz, _ = probs.size()
            
        probs = probs.view(bsz * tsz, -1)
        confid = probs[
            torch.arange(bsz * tsz),
            labels.view(bsz * tsz)
        ]
        confid  = confid.view(bsz, tsz)
        
        return confid

class ESPnetStatistic(torch.nn.Module):
    def __init__(
        self,
        ignore_id,
        bins=100,
    ):
        super().__init__()
        self.ignore_id = ignore_id

        self.bins = bins
        self.confid_hist = torch.nn.Parameter(torch.Tensor(self.bins))
        # self.register_buffer('confid_mean', torch.Tensor(1))

    @torch.no_grad()
    def forward(
        self,
        decoder_out,
        ys_out_pad,
        ys_pad_lens,
    ):
        decoder_out_prob = torch.softmax(decoder_out, dim=-1)
        confid = calc_confidence(decoder_out_prob, ys_out_pad)
        confid = [conf[:l] for conf, l in zip(confid, ys_pad_lens)]
        confid = torch.cat(confid)

        # Caculate the statistics
        # confid_mean = decoder_out_att_prob.mean()
        grad = torch.tensor([0] * self.bins).to(self.confid_hist.device)
        for i in range(self.bins):
            upper_mask = confid > i / self.bins
            lower_mask = confid < i+1 / self.bins
            grad[i] = (upper_mask * lower_mask).sum()
        self.confid_hist.grad = grad.type(self.confid_hist.dtype)
    
    def backward(
        self,
    ):
        d_p = self.confid_hist.grad

        self.confid_hist.add_(d_p, alpha=1)
        self.confid_hist.grad.detach_()
        self.confid_hist.grad.zero_()


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
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        lm: Optional[AbsLM],
        meta_encoder: Optional[AbsEncoder],
        meta_decoder: Optional[AbsDecoder],
        stat: bool = False, 
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
        # Save the results of the posterior
        self.stat = ESPnetStatistic(
            ignore_id=ignore_id
        )
        self.th = 0.10
        # Pre-trained ASR encoder-decoder archictecture
        self.meta_encoder = meta_encoder
        self.meta_decoder = meta_decoder
        # Language model
        self.lm = lm

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        noisy_label_flag: bool=False,
        replace_label_flag: bool=True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # if self.stat is not None :
        #     hist = self.stat.confid_hist
        #     if hist.sum() != 0:
        #         hist.requires_grad = False
        #         total_sum = hist.sum()
        #         # simaple mean testing for alpha = 0.27
        #         z_alpha = 18
        #         self.th = 1 / self.stat.bins * z_alpha
        #     else:
        #         # logging.warning("Prior histogram has {} value!".format(hist.sum()))
        #         self.th = None

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
            
        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None, None
        else:
            if replace_label_flag:
                decoder_meta_out_prob = self._meta_forward(
                    speech,
                    speech_lengths,
                    text,
                    text_lengths
                )
            else:
                decoder_meta_out_prob = None

            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths, 
                replace_label_flag, 
                decoder_meta_out_prob
            )
            
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
            # pred_err_att=pred_err_att,
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
        replace_label_flag: bool=False,
        decoder_out_prob=None,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)    
        ys_in_lens = ys_pad_lens + 1

        # Replace the labels
        if replace_label_flag:
            assert decoder_out_prob is not None
            from espnet.nets.pytorch_backend.nets_utils import pad_list    
            confid = calc_confidence(decoder_out_prob, ys_out_pad)

            # Eliminate the <eos> token and find the position what we replace  
            repl_mask = [prob[:l] < self.th for prob, l in zip(confid, ys_pad_lens)]

            with torch.no_grad():
                decoder_out, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
                )
                decoder_out_prob = torch.softmax(decoder_out, dim=-1).detach()

            ys_in = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            for i, (rm, y) in enumerate(zip(repl_mask, ys_in)):
                weight = decoder_out_prob[i]
                weight = weight[:len(y)]
                samples = torch.multinomial(weight, 1).squeeze(-1)
                y[rm] = samples[rm]
            _sos = ys_pad.new([self.sos])
            ys_in = [torch.cat([_sos, y], dim=0) for y in ys_in]
            ys_in_pad = pad_list(ys_in, self.eos).detach()
            
            ys_out = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            for i, (rm, y) in enumerate(zip(repl_mask, ys_out)):
                weight = decoder_out_prob[i]
                weight = weight[:len(y)]
                samples = torch.multinomial(weight, 1).squeeze(-1)
                y[rm] = samples[rm]
            # _ignore = ys_pad.new([self.ignore_id])
            _ignore = ys_pad.new([self.eos])
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys_out]
            ys_out_pad = pad_list(ys_out, self.ignore_id).detach()

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
        # if replace_label_flag:
        #     num_repl = 0.0
        #     num_total = 0.0
        #     for m in repl_mask:
        #         num_repl += m.sum()
        #         num_total += len(m)
        #     pred_err_att = float(num_repl) / float(num_total)
        # else:
        #     pred_err_att = 0.0

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

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

    ###########################################################
    ############     Meta-loop related methods     ############
    ###########################################################
    
    def _meta_collect_stats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        decoder_out, _ = self.decoder(encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)

        # Prior statistic caculator
        self.stat(decoder_out, ys_out_pad, ys_in_lens - 1) # eliminate <eos> label by reducing the ys_pad_lens
        

    def _meta_encode(
        self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder.

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
        encoder_out, encoder_out_lens, _ = self.meta_encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    @torch.no_grad()
    def _meta_forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        lam=0.6,
    ):
        assert self.meta_encoder is not None
        assert self.meta_decoder is not None
        ys_in_pad, _ = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        ys_in_lens = text_lengths + 1

        encoder_meta_out, encoder_meta_out_lens = self._meta_encode(speech, speech_lengths)
        decoder_meta_out, _ = self.meta_decoder(encoder_meta_out, encoder_meta_out_lens, ys_in_pad, ys_in_lens)

        decoder_out_prob = torch.softmax(decoder_meta_out, dim=-1)
        if self.lm is not None:
            lm_out, _ = self.lm(ys_in_pad, None)
            lm_out_prob = torch.softmax(lm_out, dim=-1)

            decoder_out_prob = lam * lm_out_prob + decoder_out_prob
        
        return decoder_out_prob