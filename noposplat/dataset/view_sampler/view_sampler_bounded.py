from dataclasses import dataclass
from typing import Literal

import random
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from noposplat.dataset.view_sampler.view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
            self,
            scene: str,
            extrinsics: Float[Tensor, "view 4 4"],
            intrinsics: Float[Tensor, "view 3 3"],
            device: torch.device = torch.device("cpu"),
            ) -> tuple[
                Int64[Tensor, " context_view"],  # indices for context views
                Int64[Tensor, " target_view"],  # indices for target views
                Float[Tensor, " overlap"],  # overlap
                ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames! in this configuration")
            
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device
            ).item()

        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        # Overfitting scenario, always pick the same indices
        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        available_views = (index_context_right - index_context_left + 1 - 2 * self.cfg.min_distance_to_context_views)

        if available_views < self.cfg.num_target_views:
            raise ValueError(f"Not enough views between context views. Required views are {self.cfg.num_target_views}, but available views are {available_views}.")

        def generate_unique_random_numbers(lb, ub, c):
            if c > (ub - lb + 1):
                raise ValueError("Cannot generate more unique random numbers than the range.")
            return sorted(random.sample(range(lb, ub + 1), c))
        
        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
                )
        else:
            # When training or validating (visualizing), pick at random.
            # index_target = torch.randint(
            #     index_context_left + self.cfg.min_distance_to_context_views,
            #     index_context_right + 1 - self.cfg.min_distance_to_context_views,
            #     size=(self.cfg.num_target_views,),
            #     device=device,
            # )
            # Ensure no duplicates and sorted target_view

            index_target_list = generate_unique_random_numbers(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right - self.cfg.min_distance_to_context_views,
                self.cfg.num_target_views
            )
            index_target = torch.tensor(index_target_list, device=device, dtype=torch.int64)
            

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.
        if self.cfg.num_context_views > 2:
            num_extra_views = self.cfg.num_context_views - 2
            extra_views = []
            while len(set(extra_views)) != num_extra_views:
                extra_views = torch.randint(
                    index_context_left + 1,
                    index_context_right,
                    (num_extra_views,),
                ).tolist()
        else:
            extra_views = []

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return (
            torch.tensor((index_context_left, *extra_views, index_context_right)),
            index_target,
            overlap
            )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
