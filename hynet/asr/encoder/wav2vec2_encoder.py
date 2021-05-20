# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class FairSeqWav2VecCtc(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.
    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        w2v_url: str,
        w2v_dir_path: str,
        input_size: int,
        output_size: int,
        apply_mask: bool,
        mask_prob: float,
        mask_channel_prob: float,
        mask_channel_length: int,
        layerdrop: float,
        activation_dropout: float,
        feature_grad_mult: float,
        freeze_finetune_updates: int,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
                # from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
                )
                raise e
        # Configuration of the model
        model.final_proj = None
        model.cfg.mask_prob = mask_prob
        # model.cfg.mask_selection = mask_selection
        # model.cfg.mask_other = mask_other
        # model.cfg.mask_length = mask_length
        # model.cfg.no_mask_overlap = no_mask_overlap
        # model.cfg.mask_min_space = mask_min_space

        model.cfg.mask_channel_prob = mask_channel_prob
        # model.cfg.mask_channel_selection = mask_channel_selection
        # model.cfg.mask_channel_other = mask_channel_other
        model.cfg.mask_channel_length = mask_channel_length
        # model.cfg.no_mask_channel_overlap = no_mask_channel_overlap
        # model.cfg.mask_channel_min_space = mask_channel_min_space

        model.cfg.encoder_layerdrop = layerdrop
        model.cfg.activation_dropout = activation_dropout
        model.cfg.feature_grad_mult = feature_grad_mult
        
        # Rearrange the model
        self._output_size = output_size

        self.apply_mask = apply_mask

        self.encoders = model
        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
        )
        
        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.
        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")

        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                masks,
                mask=self.training and self.apply_mask,
                features_only=True,
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)

        xs_pad = self.output_layer(xs_pad)

        olens = (~masks).sum(dim=1)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Wav2Vec model downloaded {model_path}")
        else:
            logging.info(f"Wav2Vec model {model_path} already exists.")

    return model_path
