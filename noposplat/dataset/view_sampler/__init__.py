from typing import Any

from noposplat.misc.step_tracker import StepTracker
from noposplat.dataset.types import Stage
from noposplat.dataset.view_sampler.view_sampler import ViewSampler
from noposplat.dataset.view_sampler.view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from noposplat.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from noposplat.dataset.view_sampler.view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from noposplat.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
