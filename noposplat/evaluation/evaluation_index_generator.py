import json
import torch.nn.functional as F
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from tqdm import tqdm

from noposplat.geometry.epipolar_lines import project_rays
from noposplat.geometry.projection import get_world_rays, sample_image_grid
from noposplat.misc.image_io import save_image
from noposplat.visualization.annotation import add_label
from noposplat.visualization.layout import add_border, hcat


@dataclass
class EvaluationIndexGeneratorCfg:
    num_target_views: int
    min_distance: int
    max_distance: int
    min_overlap: float
    max_overlap: float
    output_path: Path
    save_previews: bool
    seed: int


@dataclass
class IndexEntry:
    context: tuple[int, int]
    target: tuple[int, ...]
    overlap: Optional[str | float] = None  # choose from ["small", "medium", "large"] or a float number indicates the overlap ratio
    all_indices: Optional[tuple[int, ...]] = None



class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, IndexEntry | None]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}

    def test_step(self, batch, batch_idx):
        print("Process on batch idx: ",batch_idx)
        b, v, _, h, w = batch["target"]["image"].shape
        print("v is :", v) # v is number of frames of the seq
        assert b == 1
        extrinsics = batch["target"]["extrinsics"][0] # f*4*4
        intrinsics = batch["target"]["intrinsics"][0] # f*3*3
        scene = batch["scene"][0]

        context_indices = torch.randperm(v, generator=self.generator)
        for context_index in tqdm(context_indices, "Finding context pair"):
            print("current context_index: ", context_index)
            xy, _ = sample_image_grid((h, w), self.device)
            
            context_origins, context_directions = get_world_rays(
                rearrange(xy, "h w xy -> (h w) xy"),
                extrinsics[context_index],
                intrinsics[context_index],
            )

            # Step away from context view until the minimum overlap threshold is met.
            valid_indices = []
            for step in (1, -1):
                print("working on valid step: ", step)
                min_distance = self.cfg.min_distance
                max_distance = self.cfg.max_distance
                current_index = context_index + step * min_distance

                while 0 <= current_index.item() < v:
                    # Compute overlap.
                    current_origins, current_directions = get_world_rays(
                        rearrange(xy, "h w xy -> (h w) xy"),
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_current = project_rays(
                        context_origins,
                        context_directions,
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_context = project_rays(
                        current_origins,
                        current_directions,
                        extrinsics[context_index],
                        intrinsics[context_index],
                    )
                    overlap_a = projection_onto_context["overlaps_image"].float().mean()
                    overlap_b = projection_onto_current["overlaps_image"].float().mean()

                    overlap = min(overlap_a, overlap_b)
                    delta = (current_index - context_index).abs()

                    min_overlap = self.cfg.min_overlap
                    max_overlap = self.cfg.max_overlap
                    if min_overlap <= overlap <= max_overlap:
                        valid_indices.append(
                            (current_index.item(), overlap_a, overlap_b)
                        )

                    # Stop once the camera has panned away too much.
                    if overlap < min_overlap or delta > max_distance:
                        break

                    current_index += step

            if valid_indices:
                # Pick a random valid view. Index the resulting views.
                num_options = len(valid_indices)
                chosen = torch.randint(
                    0, num_options, size=tuple(), generator=self.generator
                )
                chosen, overlap_a, overlap_b = valid_indices[chosen]

                context_left = min(chosen, context_index.item())
                context_right = max(chosen, context_index.item())
                delta = context_right - context_left

                # Pick non-repeated random target views.
                if (context_right - context_left + 1) < self.cfg.num_target_views:
                    # Handle the error or log a message, as it's impossible to pick enough unique views
                    print("Not enough views available in the range.")
                    break
                retries = 200  # or any reasonable number
                while retries > 0:
                    target_views = torch.randint(
                        context_left,
                        context_right + 1,
                        (self.cfg.num_target_views,),
                        generator=self.generator,
                    )
                    if (target_views.unique(return_counts=True)[1] == 1).all():
                        break
                    retries -= 1
                    print("Retrying due to non-unique target views...")

                if retries == 0:
                    print("Failed to generate unique target views.")
                    break

                target = tuple(sorted(target_views.tolist()))
                self.index[scene] = IndexEntry(
                    context=(context_left, context_right),
                    target=target,
                )

                # Optionally, save a preview.
                if self.cfg.save_previews:
                    preview_path = self.cfg.output_path / "previews"
                    preview_path.mkdir(exist_ok=True, parents=True)
                    a = batch["target"]["image"][0, chosen]
                    a = add_label(a, f"Overlap: {overlap_a * 100:.1f}%")
                    b = batch["target"]["image"][0, context_index]
                    b = add_label(b, f"Overlap: {overlap_b * 100:.1f}%")
                    vis = add_border(add_border(hcat(a, b)), 1, 0)
                    vis = add_label(vis, f"Distance: {delta} frames")
                    save_image(add_border(vis), preview_path / f"{scene}.png")
                break
        else:
            # This happens if no starting frame produces a valid evaluation example.
            self.index[scene] = None


    # def generate_json(self, batch, batch_idx, num=10):
    #     print("Generating JSON for batch idx: ", batch_idx)
    #     stop_generate = True if len(self.index) >= num else False

    #     b, v, _, h, w = batch["target"]["image"].shape
    #     assert b == 1, "Batch size should be 1 for this function."
    #     extrinsics = batch["target"]["extrinsics"][0]  # f*4*4
    #     intrinsics = batch["target"]["intrinsics"][0]  # f*3*3
    #     scene = batch["scene"][0]

    #     # Step 1: Check if sequence length is sufficient
    #     if v < self.cfg.num_target_views + 1:
    #         print(f"Skipping sequence '{scene}' due to insufficient frames.")
    #         return

    #     # Step 2: Randomly choose a context frame
    #     max_start_frame = v - self.cfg.num_target_views - 1
    #     if max_start_frame < 0:
    #         print(f"Skipping sequence '{scene}' due to insufficient frames for context.")
    #         return

    #     context_start = torch.randint(0, max_start_frame + 1, (1,), generator=self.generator).item()
    #     print(f"Chosen context starting frame: {context_start}")

    #     # Step 3: Find the second context frame with valid overlap
    #     context_end = None
    #     for current_index in range(context_start + self.cfg.num_target_views - 2, v):
    #         xy, _ = sample_image_grid((h, w), self.device)
    #         context_origins, context_directions = get_world_rays(
    #             rearrange(xy, "h w xy -> (h w) xy"),
    #             extrinsics[context_start],
    #             intrinsics[context_start],
    #         )

    #         current_origins, current_directions = get_world_rays(
    #             rearrange(xy, "h w xy -> (h w) xy"),
    #             extrinsics[current_index],
    #             intrinsics[current_index],
    #         )
    #         projection_onto_current = project_rays(
    #             context_origins,
    #             context_directions,
    #             extrinsics[current_index],
    #             intrinsics[current_index],
    #         )
    #         projection_onto_context = project_rays(
    #             current_origins,
    #             current_directions,
    #             extrinsics[context_start],
    #             intrinsics[context_start],
    #         )
    #         overlap_a = projection_onto_context["overlaps_image"].float().mean()
    #         overlap_b = projection_onto_current["overlaps_image"].float().mean()
    #         overlap = min(overlap_a, overlap_b)

    #         if self.cfg.min_overlap <= overlap <= self.cfg.max_overlap:
    #             context_end = current_index
    #             print(f"Found valid context frame pair: {context_start}, {context_end} with overlap: {overlap:.2f}")
    #             break

    #     if context_end is None:
    #         print(f"No valid context pair found for sequence '{scene}'.")
    #         return

    #     # Step 4: Sample target frames between context_start and context_end
    #     valid_targets = list(range(context_start + 1, context_end))
    #     if len(valid_targets) < (self.cfg.num_target_views - 2):  # Minus 2 for the context frames
    #         print(f"Not enough valid target frames for sequence '{scene}'.")
    #         return

    #     target_indices = torch.randperm(len(valid_targets), generator=self.generator)[:(self.cfg.num_target_views - 2)]
    #     target_frames = sorted([valid_targets[idx] for idx in target_indices])

    #     # Include context frames in the output sequence
    #     all_indices = [context_start] + target_frames + [context_end]
    #     print(f"Chosen frames (sorted, including context): {all_indices}")
        
    #     # Step 5: Save results
    #     self.index[scene] = IndexEntry(
    #         context=(context_start, context_end),
    #         target=tuple(all_indices),
    #         overlap=overlap.item()  # Added to explicitly include in output if necessary
    #     )
    #     # Optionally, save individual frames as previews.
    #     if self.cfg.save_previews:
    #         # Create directory structure: <output_path>/<scene>/previews/
    #         scene_path = Path(self.cfg.output_path) / scene / "previews"
    #         scene_path.mkdir(parents=True, exist_ok=True)

    #         # Save the first context frame
    #         a = batch["target"]["image"][0, context_start]
    #         save_image(a, scene_path / f"context_frame_{context_start}.png")

    #         # Save the second context frame

    #         b = batch["target"]["image"][0, context_end]
    #         save_image(b, scene_path / f"context_frame_{context_end}.png")

    #         a = add_label(a, f"Overlap: {overlap_a * 100:.1f}%")
    #         b = add_label(b, f"Overlap: {overlap_b * 100:.1f}%")
    #         vis = add_border(add_border(hcat(a, b)), 1, 0)
    #         vis = add_label(vis, f"Distance: {abs(context_start - context_end)} frames")

    #         # Save the combined preview image
    #         save_image(add_border(vis), scene_path / f"preview_context_{context_start}_{context_end}.png")

    #     return stop_generate

    def generate_json_arbitrary(self, batch, batch_idx, num=10):
        # Step 1: Check if we need to stop generation
        if len(self.index) >= num:
            return True  # No need to continue generating if we've reached the desired number
        
        print(f"Generating JSON for batch idx: {batch_idx}")
        
        # Step 2: Extract and validate batch data
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1, "Batch size should be 1 for this function."
        scene = batch["scene"][0]

        # Step 3: Calculate camera overlap between context frames
        extrinsics_first = batch["context"]["extrinsics"][0, 0]
        intrinsics_first = batch["context"]["intrinsics"][0, 0]
        extrinsics_last = batch["context"]["extrinsics"][0, -1]
        intrinsics_last = batch["context"]["intrinsics"][0, -1]

        xy, _ = sample_image_grid((h, w), self.device)

        first_origins, first_directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            extrinsics_first,
            intrinsics_first,
        )

        last_origins, last_directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            extrinsics_last,
            intrinsics_last,
        )

        projection_onto_last = project_rays(
            first_origins,
            first_directions,
            extrinsics_last,
            intrinsics_last,
        )

        projection_onto_first = project_rays(
            last_origins,
            last_directions,
            extrinsics_first,
            intrinsics_first,
        )

        overlap_first_to_last = projection_onto_last["overlaps_image"].float().mean()
        overlap_last_to_first = projection_onto_first["overlaps_image"].float().mean()

        overlap = min(overlap_first_to_last, overlap_last_to_first)

        # Step 4: Check target trajectory validity (i.e. avoid linear motion)
        target_extrinsics = batch["target"]["extrinsics"][0]  # [T, 4, 4]

        start_pose = target_extrinsics[0]
        mid_pose = target_extrinsics[len(target_extrinsics) // 2]  # middle target frame
        end_pose = target_extrinsics[-1]                     # last frame

        if not is_valid_trajectory(start_pose, mid_pose, end_pose):
            print(f"Skipped scene {scene} due to linear or insufficient trajectory.")
            return False

        # Step 5: Prepare and store index
        target_frames = batch["target"]["index"][0].tolist()
        context_start = batch["context"]["index"][0, 0].item()
        context_end = batch["context"]["index"][0, -1].item()
        
        print(f"Context: {context_start, context_end}, Chosen frames: {target_frames}, Length: {len(target_frames)}")
        
        self.index[scene] = IndexEntry(
            context=(context_start, context_end),
            target=tuple(target_frames),
            overlap=overlap.item()
        )

        # Step 6: Optionally save preview images
        if self.cfg.save_previews:
            # Create directory structure: <output_path>/<scene>/previews/
            scene_path = Path(self.cfg.output_path) / scene / "previews"
            scene_path.mkdir(parents=True, exist_ok=True)

            # Save the first context frame
            a = batch["context"]["image"][0, 0]
            save_image(a, scene_path / f"context_frame_{context_start}.png")

            # Save the second context frame
            b = batch["context"]["image"][0, -1]
            save_image(b, scene_path / f"context_frame_{context_end}.png")

            a = add_label(a, f"Overlap: {overlap_first_to_last * 100:.1f}%")
            b = add_label(b, f"Overlap: {overlap_last_to_first * 100:.1f}%")
            vis = add_border(add_border(hcat(a, b)), 1, 0)
            vis = add_label(vis, f"Distance: {abs(context_start - context_end)} frames")

            # Save the combined preview image
            save_image(add_border(vis), scene_path / f"preview_context_{context_start}_{context_end}.png")

        return False  # Continue generating if stop_gen wasn't triggered


    def save_index(self) -> None:
        # Convert output_path to a Path object if it's a string
        output_path = Path(self.cfg.output_path)
        print(f"saving index to : {output_path}")
        output_path.mkdir(exist_ok=True, parents=True)
        json_file = output_path / "evaluation_index.json"
        with json_file.open("w") as f:
            json.dump(
                {k: None if v is None else asdict(v) for k, v in self.index.items()}, f
            )
        print(f"Saved evaluation index to {json_file}")


def is_valid_trajectory(start_pose: torch.Tensor, mid_pose: torch.Tensor, end_pose: torch.Tensor,
                        linearity_threshold: float = 0.98, min_displacement: float = 0.05) -> bool:
    start_pos = start_pose[:3, 3]
    mid_pos = mid_pose[:3, 3]
    end_pos = end_pose[:3, 3]

    displacement = (end_pos - start_pos).norm().item()
    if displacement < min_displacement:
        return False

    # 共线性判断
    v1 = mid_pos - start_pos
    v2 = end_pos - start_pos
    cos_sim = F.cosine_similarity(v1, v2, dim=0).item()

    if abs(cos_sim) > linearity_threshold:  # e.g. 0.98 means near-perfect line
        return False

    return True