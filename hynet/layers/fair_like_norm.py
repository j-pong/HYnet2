from typing import Tuple

from espnet2.layers.abs_normalize import AbsNormalize

import torch
import torch.nn.functional as F

class FairNormalize(AbsNormalize):
    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        x = F.layer_norm(x, x.shape)

        return x, ilens
