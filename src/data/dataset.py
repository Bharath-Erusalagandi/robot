"""Dataset preparation for VLA fine-tuning.

Converts DishSpace grasp annotations into (image, instruction, action) tuples
compatible with π₀ or any VLA model that expects vision-language-action inputs.

Three data sources:
  1. MuJoCo-rendered synthetic scenes (primary for bootstrap)
  2. Real robot demonstrations (from ROS bags / pilot customers)
  3. YouTube frames with pseudo-labels (supplementary)

The dataset is stored as a standard HuggingFace Dataset or a simple
list of dicts for direct use with the Trainer API.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Default to OSMesa on headless servers. It is slower than EGL but more reliable
# across container images and avoids X11/GLFW failures when EGL is unavailable.
# Must be set before mujoco is imported anywhere in the process.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

from src.models.schemas import (
    GraspAnnotation,
    ObjectType,
    EnvironmentConditions,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Language instructions for each object + condition ──

_OBJECT_INSTRUCTIONS: dict[str, list[str]] = {
    "mug": [
        "Pick up the mug by its handle",
        "Grasp the mug and move it to the drying rack",
        "Carefully lift the mug from the sink",
    ],
    "plate": [
        "Pick up the plate by its edge",
        "Grasp the plate and place it in the rack",
        "Lift the plate carefully from the counter",
    ],
    "bowl": [
        "Pick up the bowl by its rim",
        "Grasp the bowl and move it to the drying area",
        "Lift the bowl from the sink",
    ],
    "wine_glass": [
        "Carefully pick up the wine glass by its stem",
        "Gently grasp the wine glass and move it",
        "Lift the wine glass from the counter with care",
    ],
    "tumbler": [
        "Pick up the glass tumbler",
        "Grasp the tumbler and place it on the rack",
        "Lift the tumbler from the sink",
    ],
    "fork": [
        "Pick up the fork",
        "Grasp the fork by its handle",
        "Lift the fork from the sink",
    ],
    "knife": [
        "Carefully pick up the knife by its handle",
        "Grasp the knife handle safely",
        "Lift the knife from the counter",
    ],
    "spoon": [
        "Pick up the spoon",
        "Grasp the spoon by its handle",
        "Lift the spoon from the sink",
    ],
    "pot": [
        "Pick up the pot by its handle",
        "Grasp the pot handle and lift it",
        "Move the pot to the drying area",
    ],
    "pan": [
        "Pick up the pan by its handle",
        "Grasp the pan and move it to the rack",
        "Lift the pan from the sink",
    ],
    "other": [
        "Pick up this kitchen item",
        "Grasp this object and move it",
        "Lift this item from the sink",
    ],
}

_WET_MODIFIERS = [
    "The surface is wet and slippery. ",
    "Water is present on the object. ",
    "The object is wet from washing. ",
]

_SOAP_MODIFIERS = [
    "There is soap on the surface, making it extra slippery. ",
    "Soapy residue is present. Use extra grip force. ",
]

_CLUTTER_MODIFIERS = [
    "The object is in a cluttered sink scene. ",
    "Nearby dishes partially block the approach. ",
    "This grasp must avoid surrounding objects in the sink. ",
]

_PLACEMENT_MODIFIERS = {
    "drying_rack": "Place it on the drying rack after grasping. ",
    "dishwasher_top_rack": "Transfer it to the dishwasher top rack after pickup. ",
    "dishwasher_bottom_rack": "Transfer it to the dishwasher bottom rack after pickup. ",
    "utensil_caddy": "Move it into the utensil caddy after pickup. ",
}


@dataclass
class TrainingSample:
    """A single training sample for VLA fine-tuning.

    This is the format consumed by the training script.
    """

    sample_id: str
    image: np.ndarray  # RGB uint8, (H, W, 3)
    depth: Optional[np.ndarray]  # uint16 mm, (H, W)
    instruction: str  # language instruction
    action: list[float]  # 7-DOF: [x, y, z, rx, ry, rz, gripper_width_mm]
    object_type: str
    success: bool  # whether original grasp succeeded
    metadata: dict = field(default_factory=dict)


def annotation_to_instruction(
    annotation: GraspAnnotation,
    rng: np.random.Generator,
) -> str:
    """Convert a GraspAnnotation to a natural language instruction.

    Combines object-specific instruction templates with environment
    condition modifiers (wet, soapy) for diverse training language.
    """
    obj_key = annotation.object_type.value
    templates = _OBJECT_INSTRUCTIONS.get(obj_key, _OBJECT_INSTRUCTIONS["other"])
    instruction = templates[int(rng.integers(len(templates)))]

    prefix = ""
    if annotation.environment.wet:
        prefix += _WET_MODIFIERS[int(rng.integers(len(_WET_MODIFIERS)))]
    if annotation.environment.soap:
        prefix += _SOAP_MODIFIERS[int(rng.integers(len(_SOAP_MODIFIERS)))]
    if annotation.environment.visible_object_count > 3 or annotation.environment.occlusion_level > 0.3:
        prefix += _CLUTTER_MODIFIERS[int(rng.integers(len(_CLUTTER_MODIFIERS)))]

    target_zone = annotation.environment.target_zone.value
    prefix += _PLACEMENT_MODIFIERS.get(target_zone, "")

    return prefix + instruction


def annotation_to_action(annotation: GraspAnnotation) -> list[float]:
    """Convert a GraspAnnotation to a 7-DOF action vector.

    Action: [x, y, z, rx, ry, rz, gripper_width_mm]
    """
    # Grasp point xyz → workspace pose
    gp = annotation.grasp_point_xyz
    approach = annotation.approach_vector

    # Compute approach orientation as RPY
    # Default: approach from above → pitch ≈ π/2, others ≈ 0
    rx = 0.0
    ry = math.atan2(-approach[2], math.sqrt(approach[0] ** 2 + approach[1] ** 2))
    rz = math.atan2(approach[1], approach[0])

    return [
        gp[0],  # x
        gp[1],  # y
        gp[2],  # z
        rx,     # roll
        ry,     # pitch
        rz,     # yaw
        annotation.grip_width_mm,  # gripper width
    ]


def render_synthetic_image(
    annotation: GraspAnnotation,
    resolution: tuple[int, int] = (480, 640),
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an RGB-D image pair for a synthetic annotation.

    Uses MuJoCo to render a kitchen scene with the specified object,
    environment conditions, and camera pose.

    Falls back to procedural rendering if MuJoCo is not available.

    Returns:
        (rgb, depth) where rgb is uint8 (H,W,3) and depth is uint16 mm (H,W).
    """
    rng = rng or np.random.default_rng()
    h, w = resolution

    try:
        return _render_mujoco(annotation, resolution, rng)
    except ImportError:
        log.debug("mujoco_render_fallback", reason="MuJoCo not installed, using procedural")
        return _render_procedural(annotation, resolution, rng)
    except Exception as exc:
        log.warning("mujoco_render_error", error=str(exc), fallback="procedural")
        return _render_procedural(annotation, resolution, rng)


def _render_mujoco(
    annotation: GraspAnnotation,
    resolution: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Render using MuJoCo physics engine.

    Creates a scene with:
    - A kitchen sink / counter surface
    - The target object at the annotated position
    - Camera at the configured robot camera pose
    - Lighting matching the annotation conditions
    """
    import mujoco

    h, w = resolution

    # Build MJCF model for kitchen scene with target object
    xml = _build_kitchen_mjcf(annotation, rng)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=h, width=w)
    try:
        # Render RGB
        renderer.update_scene(data, camera="overhead_cam")
        rgb = renderer.render()

        # Render depth
        renderer.enable_depth_rendering()
        depth_raw = renderer.render()
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    # Convert depth from meters to uint16 mm
    depth_mm = (depth_raw * 1000).astype(np.uint16)

    # Simulate sensor artefacts for realism
    depth_mm = _add_depth_noise(depth_mm, annotation, rng)

    return rgb.copy(), depth_mm


def _build_kitchen_mjcf(
    annotation: GraspAnnotation,
    rng: np.random.Generator,
) -> str:
    """Generate MJCF XML for a kitchen scene with target object."""
    obj = annotation.object_type.value
    gp = annotation.grasp_point_xyz

    # Object geometry mapping
    geom_map = {
        "mug": f'<geom type="cylinder" size="0.04 0.05" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.8 0.3 0.2 1"/>',
        "plate": f'<geom type="cylinder" size="0.12 0.01" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.95 0.95 0.9 1"/>',
        "bowl": f'<geom type="ellipsoid" size="0.08 0.08 0.04" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.9 0.85 0.8 1"/>',
        "wine_glass": f'<geom type="cylinder" size="0.035 0.10" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.9 0.9 0.95 0.5"/>',
        "fork": f'<geom type="box" size="0.005 0.005 0.10" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.7 0.7 0.72 1"/>',
        "knife": f'<geom type="box" size="0.005 0.005 0.11" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.75 0.75 0.77 1"/>',
        "spoon": f'<geom type="capsule" size="0.01 0.08" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.72 0.72 0.74 1"/>',
        "pot": f'<geom type="cylinder" size="0.10 0.075" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.3 0.3 0.32 1"/>',
        "pan": f'<geom type="cylinder" size="0.12 0.025" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.25 0.25 0.27 1"/>',
    }
    obj_geom = geom_map.get(obj, f'<geom type="sphere" size="0.04" pos="{gp[0]} {gp[1]} {gp[2]}" rgba="0.5 0.5 0.5 1"/>')

    # Lighting based on environment
    light_intensity = "0.8" if annotation.environment.lighting == "overhead_bright" else "0.4"

    # Wet surface = darker counter, more specular
    counter_rgba = "0.35 0.35 0.38 1" if annotation.environment.wet else "0.45 0.45 0.48 1"

    return f"""
    <mujoco model="kitchen_scene">
      <visual>
        <global offwidth="640" offheight="480"/>
      </visual>
      <asset>
        <texture type="2d" name="counter_tex" builtin="flat" width="256" height="256" rgb1="0.45 0.45 0.48"/>
        <material name="counter_mat" texture="counter_tex" specular="0.3" shininess="0.1"/>
      </asset>
      <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" diffuse="{light_intensity} {light_intensity} {light_intensity}" castshadow="true"/>
        <camera name="overhead_cam" pos="0.3 0.0 0.8" xyaxes="0 -1 0 0.5 0 1" fovy="60"/>

        <!-- Counter surface -->
        <geom type="box" size="0.5 0.4 0.01" pos="0.3 0.0 0.0" rgba="{counter_rgba}" friction="0.5 0.5 0.01"/>

        <!-- Sink basin -->
        <geom type="box" size="0.2 0.15 0.08" pos="0.3 0.0 -0.04" rgba="0.7 0.7 0.72 1"/>

        <!-- Target object -->
        <body name="target_object" pos="0 0 0">
          {obj_geom}
        </body>
      </worldbody>
    </mujoco>
    """


def _add_depth_noise(
    depth: np.ndarray,
    annotation: GraspAnnotation,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add realistic depth sensor noise.

    Simulates:
    - Random holes (depth=0) at specular / transparent regions
    - Gaussian noise on valid pixels
    - Edge holes near object boundaries
    """
    result = depth.copy()

    # Gaussian noise on valid pixels (±3mm)
    valid = result > 0
    noise = rng.normal(0, 3, size=result.shape).astype(np.int16)
    result[valid] = np.clip(result[valid].astype(np.int32) + noise[valid], 1, 65535).astype(np.uint16)

    # Random holes for glass/metal objects
    mat = annotation.object_material.value
    if "glass" in mat or "metal" in mat:
        hole_rate = 0.25 if annotation.environment.wet else 0.15
        holes = rng.random(result.shape) < hole_rate
        result[holes & valid] = 0

    # Wet surface adds more holes
    if annotation.environment.wet:
        wet_holes = rng.random(result.shape) < 0.05
        result[wet_holes & valid] = 0

    return result


def _render_procedural(
    annotation: GraspAnnotation,
    resolution: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple procedural rendering when MuJoCo is not available.

    Creates a synthetic image with:
    - Grey background (counter surface)
    - Object silhouette at annotated position
    - Depth map with object at correct distance
    """
    h, w = resolution

    # Background
    bg_intensity = rng.integers(80, 140)
    rgb = np.full((h, w, 3), bg_intensity, dtype=np.uint8)

    # Object position → pixel coordinates
    gp = annotation.grasp_point_xyz
    cx = int(np.clip((gp[0] - 0.1) / 0.5 * w, 50, w - 50))
    cy = int(np.clip((gp[1] + 0.2) / 0.4 * h, 50, h - 50))

    # Object colour by type
    colours = {
        "mug": (180, 80, 60),
        "plate": (230, 225, 215),
        "bowl": (220, 210, 195),
        "wine_glass": (200, 200, 210),
        "fork": (170, 170, 175),
        "knife": (175, 175, 180),
        "spoon": (168, 168, 172),
        "pot": (75, 75, 80),
        "pan": (60, 60, 65),
    }
    colour = colours.get(annotation.object_type.value, (120, 120, 120))

    # Draw object as ellipse
    obj_w = rng.integers(30, 80)
    obj_h = rng.integers(30, 80)
    y_coords, x_coords = np.ogrid[-obj_h:obj_h, -obj_w:obj_w]
    mask = (x_coords ** 2 / max(obj_w ** 2, 1) + y_coords ** 2 / max(obj_h ** 2, 1)) <= 1.0

    y_start = max(0, cy - obj_h)
    y_end = min(h, cy + obj_h)
    x_start = max(0, cx - obj_w)
    x_end = min(w, cx + obj_w)

    mask_cropped = mask[
        (y_start - cy + obj_h):(y_end - cy + obj_h),
        (x_start - cx + obj_w):(x_end - cx + obj_w),
    ]

    for c in range(3):
        region = rgb[y_start:y_end, x_start:x_end, c]
        region[mask_cropped] = colour[c]

    # Depth map
    base_depth = max(int(gp[2] * 1000), 300)  # mm
    depth = np.full((h, w), base_depth + 200, dtype=np.uint16)  # background
    depth[y_start:y_end, x_start:x_end][mask_cropped] = base_depth

    # Add noise
    depth = _add_depth_noise(depth, annotation, rng)

    return rgb, depth


def prepare_training_dataset(
    annotations_path: str | Path,
    output_dir: str | Path,
    max_samples: Optional[int] = None,
    render: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Prepare a full training dataset from annotations JSON.

    Reads GraspAnnotation records, generates images + language instructions,
    and produces a list of training dicts ready for the Trainer.

    Args:
        annotations_path: Path to annotations JSON file.
        output_dir: Directory to save rendered images.
        max_samples: Limit number of samples (for quick testing).
        render: Whether to render images (False = metadata only).
        seed: Random seed.

    Returns:
        List of training sample dicts with keys:
        - image_path, depth_path, instruction, action, object_type, success
    """
    annotations_path = Path(annotations_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with open(annotations_path) as f:
        raw = json.load(f)

    annotations = [GraspAnnotation(**r) for r in raw]
    if max_samples:
        annotations = annotations[:max_samples]

    samples: list[dict] = []

    for i, ann in enumerate(annotations):
        instruction = annotation_to_instruction(ann, rng)
        action = annotation_to_action(ann)

        sample = {
            "sample_id": ann.sample_id,
            "instruction": instruction,
            "action": action,
            "object_type": ann.object_type.value,
            "success": ann.success,
            "wet": ann.environment.wet,
            "soap": ann.environment.soap,
            "material": ann.object_material.value,
            "failure_mode": ann.failure_mode.value,
            "scene_type": ann.environment.scene_type.value,
            "visible_object_count": ann.environment.visible_object_count,
            "occlusion_level": ann.environment.occlusion_level,
            "target_zone": ann.environment.target_zone.value,
        }

        if render:
            rgb, depth = render_synthetic_image(ann, rng=rng)

            rgb_path = output_dir / f"{ann.sample_id}_rgb.npy"
            depth_path = output_dir / f"{ann.sample_id}_depth.npy"

            np.save(rgb_path, rgb)
            np.save(depth_path, depth)

            sample["image_path"] = str(rgb_path)
            sample["depth_path"] = str(depth_path)

        samples.append(sample)

        if (i + 1) % 500 == 0:
            log.info("dataset_prep_progress", prepared=i + 1, total=len(annotations))

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    # Strip numpy arrays for JSON serialisation
    json_samples = [
        {k: v for k, v in s.items() if not isinstance(v, np.ndarray)}
        for s in samples
    ]
    manifest_path.write_text(json.dumps(json_samples, indent=2))

    # Stats
    success_rate = sum(1 for s in samples if s["success"]) / max(len(samples), 1)
    wet_rate = sum(1 for s in samples if s.get("wet")) / max(len(samples), 1)

    log.info(
        "dataset_prepared",
        total=len(samples),
        success_rate=f"{success_rate:.1%}",
        wet_pct=f"{wet_rate:.1%}",
        output=str(output_dir),
    )

    return samples


class GraspDataset:
    """PyTorch-compatible dataset for VLA fine-tuning.

    Lazily loads images and returns (image, instruction, action) tuples
    formatted for the model's processor.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        processor=None,
        max_length: int = 128,
        success_only: bool | None = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.processor = processor
        self.max_length = max_length

        with open(self.manifest_path) as f:
            self.samples = json.load(f)

        manifest_dir = self.manifest_path.parent
        for sample in self.samples:
            for key in ("image_path", "depth_path"):
                path_value = sample.get(key)
                if path_value and not Path(path_value).is_absolute():
                    sample[key] = str((manifest_dir / path_value).resolve())

        if success_only:
            self.samples = [s for s in self.samples if s.get("success", False)]

        log.info("grasp_dataset_loaded", samples=len(self.samples), path=str(manifest_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        result = {
            "instruction": sample["instruction"],
            "action": sample["action"],
            "object_type": sample["object_type"],
            "success": sample.get("success", False),
            "failure_mode": sample.get("failure_mode", "none"),
            "scene_type": sample.get("scene_type", "sink_single"),
            "visible_object_count": sample.get("visible_object_count", 1),
            "occlusion_level": sample.get("occlusion_level", 0.0),
            "target_zone": sample.get("target_zone", "drying_rack"),
        }

        # Load image if path exists
        if "image_path" in sample and Path(sample["image_path"]).exists():
            rgb = np.load(sample["image_path"])
            result["image"] = rgb

        if "depth_path" in sample and Path(sample["depth_path"]).exists():
            depth = np.load(sample["depth_path"])
            result["depth"] = depth

        # If we have a processor, tokenise
        if self.processor and "image" not in result:
            raise RuntimeError(
                f"Sample {sample.get('sample_id', idx)} is missing a rendered image. "
                "Training requires render=True so pixel inputs exist."
            )

        if self.processor and "image" in result:
            from PIL import Image

            pil_img = Image.fromarray(result["image"])
            try:
                encoded = self.processor(
                    images=pil_img,
                    text=result["instruction"],
                    return_tensors="pt",
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                )
            except TypeError as exc:
                if "max_length" not in str(exc):
                    raise RuntimeError(
                        f"Processor failed for sample {sample.get('sample_id', idx)}: {exc}"
                    ) from exc
                encoded = self.processor(
                    images=pil_img,
                    text=result["instruction"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Processor failed for sample {sample.get('sample_id', idx)}: {exc}"
                ) from exc
            result["pixel_values"] = encoded["pixel_values"].squeeze(0)
            if "input_ids" in encoded:
                result["input_ids"] = encoded["input_ids"].squeeze(0)
                result["attention_mask"] = encoded["attention_mask"].squeeze(0)

        return result

    def train_test_split(
        self,
        test_pct: float = 0.15,
        seed: int = 42,
    ) -> tuple["GraspDataset", "GraspDataset"]:
        """Split into train and test sets."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.samples))
        n_test = int(len(self.samples) * test_pct)

        test_samples = [self.samples[i] for i in indices[:n_test]]
        train_samples = [self.samples[i] for i in indices[n_test:]]

        train_ds = GraspDataset.__new__(GraspDataset)
        train_ds.manifest_path = self.manifest_path
        train_ds.processor = self.processor
        train_ds.max_length = self.max_length
        train_ds.samples = train_samples

        test_ds = GraspDataset.__new__(GraspDataset)
        test_ds.manifest_path = self.manifest_path
        test_ds.processor = self.processor
        test_ds.max_length = self.max_length
        test_ds.samples = test_samples

        return train_ds, test_ds
