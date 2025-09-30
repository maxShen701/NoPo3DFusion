from dataclasses import dataclass
from typing import Literal

import torch
import random
from jaxtyping import Float, Int64
from torch import Tensor

from noposplat.dataset.view_sampler.three_view_hack import add_third_context_index
from noposplat.dataset.view_sampler.view_sampler import ViewSampler


@dataclass
class ViewSamplerArbitraryCfg:
    name: Literal["arbitrary"]
    num_context_views: int
    num_target_views: int
    dist_of_context_view_in: int
    consider_seq_length: int
    context_views: list[int] | None
    target_views: list[int] | None


class ViewSamplerArbitrary(ViewSampler[ViewSamplerArbitraryCfg]):
    """
    Uniformly sample target views from a sequence of views.
    The context views are the first and the 13th view in the sampled target views.
    """
    def sample(
            self,
            scene: str,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            device: torch.device = torch.device("cpu"),
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Arbitrarily sample context and target views based on the new construction method."""
        num_views, _, _ = extrinsics.shape

        if num_views < self.cfg.num_target_views:
            raise ValueError(f"num_views ({num_views}) is less than num_target_views ({self.cfg.num_target_views}).")
        
        step, subseq_length = closest_subsequence_length(self.cfg.num_target_views, num_views)
        start_index = torch.randint(0, num_views - subseq_length + 1, (1,), device=device).item()
        end_index = start_index + subseq_length

        index_target_subseq = torch.arange(start_index, end_index, step, device=device).round().long()

        if len(index_target_subseq) != self.cfg.num_target_views:
            index_target_subseq = index_target_subseq[:self.cfg.num_target_views]

        context_1_index = index_target_subseq[0]
        context_2_index = index_target_subseq[12]

        index_target = index_target_subseq.sort()[0]
        index_context = torch.tensor([context_1_index, context_2_index], dtype=torch.int64, device=device)

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return index_context, index_target, overlap
        

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


def closest_subsequence_length(k, seq_length):
    m = seq_length // k
    closest_multiple = m * k
    return m, closest_multiple