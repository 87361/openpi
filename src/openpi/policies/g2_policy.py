import dataclasses

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
class G2Inputs(transforms.DataTransformFn):
    """Inputs for Kuavo G2 datasets with head and two hand cameras."""

    def __call__(self, data: dict) -> dict:
        images = data["images"]

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": _parse_image(images["cam_high"]),
                "left_wrist_0_rgb": _parse_image(images["cam_left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["cam_right_wrist"]),
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
class G2Outputs(transforms.DataTransformFn):
    """Drop pi05 padding dimensions and return native G2/Kuavo actions."""

    action_dim: int = 24

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
