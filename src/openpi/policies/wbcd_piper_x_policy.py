import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WbcdPiperXInputs(transforms.DataTransformFn):
    """Inputs for active-arm WBCD Piper-X datasets with 3 cameras."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_under",
        "cam_active_wrist",
    )
    LEGACY_ACTIVE_WRIST_CAMERA: ClassVar[str] = "cam_right_wrist"

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        allowed_cameras = set(self.EXPECTED_CAMERAS) | {self.LEGACY_ACTIVE_WRIST_CAMERA}
        if set(in_images) - allowed_cameras:
            raise ValueError(f"Expected images to contain active-arm cameras, got {tuple(in_images)}")

        missing_cameras = set(self.EXPECTED_CAMERAS[:2]) - set(in_images)
        if missing_cameras:
            raise ValueError(f"Missing required cameras: {tuple(sorted(missing_cameras))}")

        active_wrist_camera = (
            "cam_active_wrist" if "cam_active_wrist" in in_images else self.LEGACY_ACTIVE_WRIST_CAMERA
        )
        if active_wrist_camera not in in_images:
            raise ValueError("Expected images to contain cam_active_wrist or cam_right_wrist")

        base_image = _parse_image(in_images["cam_high"])
        under_image = _parse_image(in_images["cam_under"])
        active_wrist_image = _parse_image(in_images[active_wrist_camera])

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": under_image,
                "right_wrist_0_rgb": active_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class WbcdPiperXOutputs(transforms.DataTransformFn):
    """Outputs for single-arm WBCD Piper-X datasets."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
