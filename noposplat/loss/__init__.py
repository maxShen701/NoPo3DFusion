from noposplat.loss.loss import Loss
from noposplat.loss.loss_depth import LossDepth, LossDepthCfgWrapper
from noposplat.loss.loss_lpips import LossLpips, LossLpipsCfgWrapper
from noposplat.loss.loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
