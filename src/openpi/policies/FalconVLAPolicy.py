import dataclasses
import time
from typing import Any, Optional

import numpy as np
import torch

from openpi_client import base_policy as _base_policy

# import your inference utilities
from openpi.models.FalconVLA import load_falcon_model, predict_actions_chunk, normalize_gripper_action, invert_gripper_action
# if you want these gripper helpers:

def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Accepts HWC or CHW; returns HWC uint8."""
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")
    # If CHW, convert -> HWC
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    # convert float -> uint8 if needed
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * np.clip(img, 0.0, 1.0)).astype(np.uint8)
    else:
        img = img.astype(np.uint8, copy=False)
    return img


@dataclasses.dataclass
class FalconVLAConfig:
    model_name: str
    hf_token: Optional[str] = None
    horizon: int = 5
    unnorm_key: str = "ee_delta"
    use_wrist: bool = False
    use_secondary: bool = False
    use_proprio: bool = False

    # Optional post-processing knobs:
    normalize_gripper: bool = False   # [0,1] -> [-1,+1] if your env needs it
    invert_gripper: bool = False      # flips sign on last dim if env convention differs


class FalconVLAPolicy(_base_policy.BasePolicy):
    def __init__(self, cfg: FalconVLAConfig, *, default_prompt: str | None = None):
        self._cfg = cfg
        self._default_prompt = default_prompt
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor, self._vla = load_falcon_model(cfg.model_name, cfg.hf_token)
        self._vla.eval()

        self._metadata: dict[str, Any] = {
            "name": "falcon_vla_policy",
            "model_name": cfg.model_name,
            "device": self._device,
            "horizon": cfg.horizon,
            "unnorm_key": cfg.unnorm_key,
            "use_wrist": cfg.use_wrist,
            "use_secondary": cfg.use_secondary,
            "use_proprio": cfg.use_proprio,
        }

    def infer(self, obs: dict) -> dict:
        """
        Expected obs (recommended OpenPI-ish shape):
          - prompt: str (optional; falls back to default_prompt)
          - images: dict with at least a base camera
              e.g. images["cam_high"] or images["base_0_rgb"]
          - optionally wrist/secondary images depending on cfg
          - optionally state if you enable use_proprio
        """
        prompt = obs.get("prompt", None) or self._default_prompt
        if not prompt:
            raise ValueError("No prompt provided (obs['prompt'] missing and no default_prompt set).")

        images = obs.get("images", None)
        if images is None or not isinstance(images, dict):
            raise ValueError("Expected obs['images'] as a dict of images.")

        # Pick a primary image key (align this with your client payload)
        primary = None
        for k in ("cam_high", "base_0_rgb", "primary_image"):
            if k in images:
                primary = images[k]
                break
        if primary is None:
            raise ValueError(f"Could not find a primary image. Got keys: {tuple(images.keys())}")

        observation = {"primary_image": _to_hwc_uint8(primary)}

        if self._cfg.use_wrist:
            wrist = None
            for k in ("cam_left_wrist", "left_wrist_0_rgb", "wrist_image"):
                if k in images:
                    wrist = images[k]
                    break
            if wrist is None:
                raise ValueError("use_wrist=True but no wrist image key found.")
            observation["wrist_image"] = _to_hwc_uint8(wrist)

        if self._cfg.use_secondary:
            secondary = None
            for k in ("cam_low", "secondary_image"):
                if k in images:
                    secondary = images[k]
                    break
            if secondary is None:
                raise ValueError("use_secondary=True but no secondary image key found.")
            observation["secondary_image"] = _to_hwc_uint8(secondary)

        # Pass state through if you enable proprio in your prompt builder
        if self._cfg.use_proprio and "state" in obs:
            observation["state"] = np.asarray(obs["state"])

        start = time.monotonic()
        actions_t = predict_actions_chunk(
            vla=self._vla,
            processor=self._processor,
            observation=observation,
            task_label=prompt,
            horizon=self._cfg.horizon,
            unnorm_key=self._cfg.unnorm_key,
            use_wrist=self._cfg.use_wrist,
            use_secondary=self._cfg.use_secondary,
            use_proprio=self._cfg.use_proprio,
        )
        
        # Convert to numpy if needed
        if isinstance(actions_t, torch.Tensor):
            actions_np = actions_t.detach().float().cpu().numpy()
        else:
            actions_np = np.asarray(actions_t, dtype=np.float32)
        
        # Apply post-processing
        if self._cfg.normalize_gripper:
            actions_np = normalize_gripper_action(actions_np, binarize=True)
        if self._cfg.invert_gripper:
            actions_np = invert_gripper_action(actions_np)
        

        # Reshape from (350,) to (25, 14)
        actions_np = actions_np.reshape(self._cfg.horizon, -1)

        # If we got 7D actions but need 14D (duplicated for bimanual), duplicate
        if actions_np.shape[-1] == 7:
            actions_np = np.concatenate([actions_np, actions_np], axis=-1)

        infer_ms = (time.monotonic() - start) * 1000.0

        return {
            "actions": actions_np,              # (H, 7) or (H, 14) after duplication
            "policy_timing": {"infer_ms": infer_ms},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata