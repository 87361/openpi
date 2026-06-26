"""WBCD Aloha policy transforms with 4-camera support.

This module extends the standard AlohaInputs to support a 4th camera (cam_low)
mapped to base_1_rgb, while keeping the original AlohaInputs untouched.
"""
import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms
from openpi.policies.aloha_policy import AlohaOutputs, _decode_aloha  # noqa: F401 – re-export


@dataclasses.dataclass(frozen=True)
class WbcdAloha4CamInputs(transforms.DataTransformFn):
    """Inputs for WBCD Aloha with 4 cameras.

    Extends the standard 3-camera layout with a 4th camera (cam_low → base_1_rgb).

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [14]
    - actions: [action_horizon, 14]
    """

    adapt_to_pi: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
            )

        # cam_high is always required as the primary base image.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Extra images: cam_low (4th cam), left/right wrist
        extra_image_names = {
            "base_1_rgb": "cam_low",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            # No adapt_to_pi encoding for WBCD
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
