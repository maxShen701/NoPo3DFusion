from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from noposplat.dataset.types import BatchedExample
from noposplat.model.decoder.decoder import DecoderOutput
from noposplat.model.types import Gaussians
from noposplat.loss.loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        return self.cfg.weight * (delta**2).mean()
