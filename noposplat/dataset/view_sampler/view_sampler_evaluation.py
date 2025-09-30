import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from noposplat.evaluation.evaluation_index_generator import IndexEntry
from noposplat.global_cfg import get_cfg
from noposplat.misc.step_tracker import StepTracker
from noposplat.dataset.types import Stage
from noposplat.dataset.view_sampler.three_view_hack import add_third_context_index
from noposplat.dataset.view_sampler.view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        self.cfg = cfg

        dacite_config = Config(cast=[tuple])

        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config) # check if v is not None, convert to IndexEntry object
                for k, v in json.load(f).items()
            }

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
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        # Handle 2-view index for 3 views.
        v = self.num_context_views
        if v > len(context_indices) and v == 3:
            context_indices = add_third_context_index(context_indices)

        return context_indices, target_indices, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0
    

import numpy as np

class FixedViewSampler(ViewSamplerEvaluation):
    def __init__(self, cfg, stage, is_overfitting, cameras_are_circular, step_tracker):
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        self.rotation_angle = 180  # l-r rotation range
        self.num_frames = 49  # number of target views on track
        self.rotation_step = self.rotation_angle / self.num_frames  

    def generate_camera_trajectory(self, initial_extrinsics):
        trajectory = []
        for frame in range(self.num_frames):
            angle = self.rotation_step * frame
            rotation_matrix = self.create_rotation_matrix(angle)

            new_extrinsics = torch.matmul(rotation_matrix, initial_extrinsics)
            trajectory.append(new_extrinsics)
        return trajectory

    def create_rotation_matrix(self, angle_deg):
        """创建绕y轴旋转的旋转矩阵"""
        angle_rad = np.radians(angle_deg)
        rotation_matrix = torch.tensor([
            [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        return rotation_matrix

    def sample(self, scene: str, 
               extrinsics: Float[Tensor, "view 4 4"], 
               intrinsics: Float[Tensor, "view 3 3"], 
               device: torch.device = torch.device("cpu")) -> tuple:
        """在固定的轨迹上进行采样"""
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        # 使用第一个context view作为初始相机位置
        initial_extrinsics = extrinsics[context_indices[0]]

        # 生成固定的相机轨迹
        trajectory = self.generate_camera_trajectory(initial_extrinsics)

        # 这里我们不改变上下文视角和目标视角的选择，仍然返回原来的数据
        overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        return context_indices, target_indices, overlap, trajectory


