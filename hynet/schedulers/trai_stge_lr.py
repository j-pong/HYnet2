import math

from distutils.version import LooseVersion
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class TriStageLR(_LRScheduler, AbsBatchStepScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr_scale: Union[int, float],
        final_lr_scale: Union[int, float],
        hold_steps: Union[int, float],
        decay_steps: Union[int, float],
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        if LooseVersion(torch.__version__) < LooseVersion("1.1.0"):
            raise NotImplementedError(f"Require PyTorch>=1.1.0: {torch.__version__}")

        assert check_argument_types()

        # calculate LR at each point
        self.peak_lr = self.base_lrs[0]
        self.init_lr = init_lr_scale * self.base_lrs[0]
        self.final_lr = final_lr_scale * self.base_lrs[0]

        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def _decide_stage(self, update_step):
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def get_lr(self):
        stage, steps_in_stage = self._decide_stage(self.last_epoch + 1)
        if stage == 0:
            lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            lr = self.peak_lr
        elif stage == 2:
            lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        return [lr]
        # step_num = self.last_epoch + 1
        # return [
        #     lr
        #     * self.warmup_steps ** 0.5
        #     * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
        #     for lr in self.base_lrs
        # ]
