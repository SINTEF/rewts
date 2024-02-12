from typing import Optional
import torch


class RMSELoss(torch.nn.MSELoss):
    def __init__(self, eps: float = 1e-8, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = "mean"):
        self.eps = eps
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.functional.F.mse_loss(input, target, reduction=self.reduction)
        return torch.sqrt(mse + self.eps)