import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

# custom
import os
from pathlib import Path

from espnet2.torch_utils.model_summary import model_summary
from espnet2.torch_utils.pytorch_version import pytorch_cudnn_version
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.types import str2triple_str
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump
from espnet2.torch_utils.load_pretrained_model import load_pretrained_model

from hynet.main_funcs.collect_stats import collect_stats
from hynet.train.trainer import Trainer
from hynet.asr.espnet_model import ESPnetASRModel
from hynet.asr.decoder.rnn_decoder import RNNDecoder
from hynet.torch_utils.initialize import initialize

from espnet2.tasks.abs_task import IteratorOptions

from espnet2.lm.abs_model import AbsLM
from espnet2.lm.espnet_model import ESPnetLanguageModel
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend, sliding_window=SlidingWindow),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)
lm_choices = ClassChoices(
    "lm",
    classes=dict(
        seq_rnn=SequentialRNNLM,
        transformer=TransformerLM,
    ),
    type_check=AbsLM,
    default=None,
    optional=True,
)


class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --lm and --lm_conf
        lm_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        required = parser.get_default("required")
        required += ["token_list"]
        
        group.add_argument("--train_pseudo_shape_file", type=str, action="append", default=[])
        
        group.add_argument(
            "--train_pseudo_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
            help="Give three words splitted by comma. It's used for the training data.",
        )
        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--stage",
            type=int,
            default=1,
            help="The train stage for SSL",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetASRModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=[None, "g2p_en", "pyopenjtalk", "pyopenjtalk_kana"],
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod  # override the AbsTask
    def build_iter_options(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
    ):
        if mode == "train":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = args.train_data_path_and_name_and_type
            shape_files = args.train_shape_file
            batch_size = args.batch_size
            batch_bins = args.batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = args.num_iters_per_epoch
            train = True

        elif mode == "pseudo":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = args.train_pseudo_data_path_and_name_and_type
            shape_files = args.train_pseudo_shape_file
            batch_size = args.batch_size
            batch_bins = args.batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = args.num_iters_per_epoch
            train = True

        elif mode == "valid":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = args.valid_data_path_and_name_and_type
            shape_files = args.valid_shape_file

            if args.valid_batch_type is None:
                batch_type = args.batch_type
            else:
                batch_type = args.valid_batch_type
            if args.valid_batch_size is None:
                batch_size = args.batch_size
            else:
                batch_size = args.valid_batch_size
            if args.valid_batch_bins is None:
                batch_bins = args.batch_bins
            else:
                batch_bins = args.valid_batch_bins
            if args.valid_max_cache_size is None:
                # Cache 5% of maximum size for validation loader
                max_cache_size = 0.05 * args.max_cache_size
            else:
                max_cache_size = args.valid_max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = None
            train = False

        elif mode == "plot_att":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = args.valid_data_path_and_name_and_type
            shape_files = args.valid_shape_file
            batch_type = "unsorted"
            batch_size = 1
            batch_bins = 0
            num_batches = args.num_att_plot
            max_cache_fd = args.max_cache_fd
            # num_att_plot should be a few sample ~ 3, so cache all data.
            max_cache_size = np.inf if args.max_cache_size != 0.0 else 0.0
            # always False because plot_attention performs on RANK0
            distributed = False
            num_iters_per_epoch = None
            train = False
        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data_path_and_name_and_type=data_path_and_name_and_type,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
        )

    @classmethod  # override the AbsTask
    def main_worker(cls, args: argparse.Namespace):
        assert check_argument_types()

        # 0. Init distributed process
        distributed_option = build_dataclass(DistributedOption, args)
        distributed_option.init()

        # NOTE(kamo): Don't use logging before invoking logging.basicConfig()
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            if not distributed_option.distributed:
                _rank = ""
            else:
                _rank = (
                    f":{distributed_option.dist_rank}/"
                    f"{distributed_option.dist_world_size}"
                )

            # NOTE(kamo):
            # logging.basicConfig() is invoked in main_worker() instead of main()
            # because it can be invoked only once in a process.
            # FIXME(kamo): Should we use logging.getLogger()?
            logging.basicConfig(
                level=args.log_level,
                format=f"[{os.uname()[1].split('.')[0]}{_rank}]"
                f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            # Suppress logging if RANK != 0
            logging.basicConfig(
                level="ERROR",
                format=f"[{os.uname()[1].split('.')[0]}"
                f":{distributed_option.dist_rank}/{distributed_option.dist_world_size}]"
                f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )

        # 1. Set random-seed
        set_all_random_seed(args.seed)
        torch.backends.cudnn.enabled = args.cudnn_enabled
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = args.cudnn_deterministic
        if args.detect_anomaly:
            logging.info("Invoking torch.autograd.set_detect_anomaly(True)")
            torch.autograd.set_detect_anomaly(args.detect_anomaly)

        # 2. Build model
        model = cls.build_model(args=args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model = model.to(
            dtype=getattr(torch, args.train_dtype),
            device="cuda" if args.ngpu > 0 else "cpu",
        )
        for t in args.freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logging.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # 3. Build optimizer
        optimizers = cls.build_optimizers(args, model=model)

        # 4. Build schedulers
        schedulers = []
        for i, optim in enumerate(optimizers, 1):
            suf = "" if i == 1 else str(i)
            name = getattr(args, f"scheduler{suf}")
            conf = getattr(args, f"scheduler{suf}_conf")
            if name is not None:
                cls_ = scheduler_classes.get(name)
                if cls_ is None:
                    raise ValueError(
                        f"must be one of {list(scheduler_classes)}: {name}"
                    )
                scheduler = cls_(optim, **conf)
            else:
                scheduler = None

            schedulers.append(scheduler)

        logging.info(pytorch_cudnn_version())
        logging.info(model_summary(model))
        for i, (o, s) in enumerate(zip(optimizers, schedulers), 1):
            suf = "" if i == 1 else str(i)
            logging.info(f"Optimizer{suf}:\n{o}")
            logging.info(f"Scheduler{suf}: {s}")

        # 5. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
                logging.info(
                    f'Saving the configuration in {output_dir / "config.yaml"}'
                )
                yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        # 6. Loads pre-trained model
        for p in args.init_param:
            logging.info(f"Loading pretrained params from {p}")
            load_pretrained_model(
                model=model,
                init_param=p,
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                map_location=f"cuda:{torch.cuda.current_device()}"
                if args.ngpu > 0
                else "cpu",
            )

        if args.dry_run:
            pass
        elif args.collect_stats:
            # Perform on collect_stats mode. This mode has two roles
            # - Derive the length and dimension of all input data
            # - Accumulate feats, square values, and the length for whitening

            if args.valid_batch_size is None:
                args.valid_batch_size = args.batch_size

            if len(args.train_shape_file) != 0:
                train_key_file = args.train_shape_file[0]
            else:
                train_key_file = None
            if len(args.train_pseudo_shape_file) != 0:
                train_pseudo_key_file = args.train_pseudo_shape_file[0]
            else:
                train_pseudo_key_file = None
            if len(args.valid_shape_file) != 0:
                valid_key_file = args.valid_shape_file[0]
            else:
                valid_key_file = None

            collect_stats(
                model=model,
                train_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.train_data_path_and_name_and_type,
                    key_file=train_key_file,
                    batch_size=args.batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                train_pseudo_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.train_pseudo_data_path_and_name_and_type,
                    key_file=train_pseudo_key_file,
                    batch_size=args.batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                valid_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
                    key_file=valid_key_file,
                    batch_size=args.valid_batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                output_dir=output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
                write_collected_feats=args.write_collected_feats,
            )
        else:

            # 7. Build iterator factories
            assert not args.multiple_iterator
            train_iter_factory = cls.build_iter_factory(
                args=args,
                distributed_option=distributed_option,
                mode="train",
            )

            train_pseudo_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="pseudo",
            )

            valid_iter_factory = cls.build_iter_factory(
                args=args,
                distributed_option=distributed_option,
                mode="valid",
            )
            if args.num_att_plot != 0:
                plot_attention_iter_factory = cls.build_iter_factory(
                    args=args,
                    distributed_option=distributed_option,
                    mode="plot_att",
                )
            else:
                plot_attention_iter_factory = None

            # 8. Start training
            if args.use_wandb:
                if (
                    not distributed_option.distributed
                    or distributed_option.dist_rank == 0
                ):
                    if args.wandb_project is None:
                        project = (
                            "ESPnet_"
                            + cls.__name__
                            + str(Path(".").resolve()).replace("/", "_")
                        )
                    else:
                        project = args.wandb_project
                    if args.wandb_id is None:
                        wandb_id = str(output_dir).replace("/", "_")
                    else:
                        wandb_id = args.wandb_id

                    wandb.init(
                        project=project,
                        dir=output_dir,
                        id=wandb_id,
                        resume="allow",
                    )
                    wandb.config.update(args)
                else:
                    # wandb also supports grouping for distributed training,
                    # but we only logs aggregated data,
                    # so it's enough to perform on rank0 node.
                    args.use_wandb = False

            # Don't give args to trainer.run() directly!!!
            # Instead of it, define "Options" object and build here.
            trainer_options = cls.trainer.build_options(args)
            cls.trainer.run(
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                train_iter_factory=train_iter_factory,
                train_pseudo_iter_factory=train_pseudo_iter_factory,
                valid_iter_factory=valid_iter_factory,
                plot_attention_iter_factory=plot_attention_iter_factory,
                trainer_options=trainer_options,
                distributed_option=distributed_option,
            )

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")
        
        # 0. language model for finding the corrupted label
        if args.lm is not None:
            lm_class = lm_choices.get_class(args.lm)
            lm = lm_class(vocab_size=vocab_size, **args.lm_conf)
        else:
            lm = None

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 6. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder.output_size(), **args.ctc_conf
        )

        # 7. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 8. Build model
        if args.stage == 3:
            encoder_class = encoder_choices.get_class(args.encoder)
            meta_encoder = encoder_class(input_size=input_size, **args.encoder_conf)
            decoder_class = decoder_choices.get_class(args.decoder)
            meta_decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder.output_size(),
                **args.decoder_conf,
            )
            model = ESPnetASRModel(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                lm=lm,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                meta_encoder=meta_encoder,
                meta_decoder=meta_decoder,
                stat=True,
                rnnt_decoder=rnnt_decoder,
                token_list=token_list,
                **args.model_conf,
            )
        else:
            model = ESPnetASRModel(
                vocab_size=vocab_size,
                frontend=frontend,
                specaug=specaug,
                normalize=normalize,
                preencoder=preencoder,
                lm=lm,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                meta_encoder=None,
                meta_decoder=None,
                stat=True,
                rnnt_decoder=rnnt_decoder,
                token_list=token_list,
                **args.model_conf,
            )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model