"""
Load and predict functions for falcon vla.
Author: Brahim Farhat
Date: 01/09/2025
Version: 1.0

© 2025 TII. All rights reserved.
"""
from typing import Tuple, Optional, Dict, Any, List
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_DIM = 7



# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99

IMAGE_BLOCK = "|<start_of_img>||<image>||<end_of_img>|"
TURN_PREFIX = "|<start_of_turn>|User: "
THINK_TAG = "<thinking>"

def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio

def tokenize_proprio(
    x: float | np.ndarray, 
    num_bins: int = 256,
) -> list[str]:
    """
    Encode float(s) in [-1, 1] into bin indices at the *end* of the vocab.
    Indices will be in [vocab_size - num_bins, vocab_size - 1].

    Args:
        x (float): or array of floats in [-1, 1].
        num_bins (int): number of bins (default 256).

    Returns:
        list[str]: tokenized array of proprioceptive states ["<prop0>", "<prop11>",...]
    """
    if np.any((x < -1.0) | (x > 1.0)):
        raise ValueError("Input values must be in [-1, 1]")

    edges = np.linspace(-1, 1, num_bins)
    rel_idx = np.digitize(x, edges, right=False) - 1
    rel_idx = rel_idx.astype(np.int64).ravel()
    return [f"<prop{int(idx)}>" for idx in rel_idx]


@dataclass
class ActionPromptBuilder:
    task_label: str
    main_image: Optional[Any] = None
    wrist_image: Optional[Any] = None
    secondary_image: Optional[Any] = None
    proprio_tokens: Optional[str] = None

    def add_main_image(self, image: Any) -> "ActionPromptBuilder":
        self.main_image = image
        return self

    def add_wrist_image(self, wrist_image: Any) -> "ActionPromptBuilder":
        self.wrist_image = wrist_image
        return self
    
    def add_secondary_image(self, secondary_image: Any) -> "ActionPromptBuilder":
        self.secondary_image = secondary_image
        return self

    def get_images(self) -> List[Any]:
        return [self.main_image, self.wrist_image, self.secondary_image]

    def add_proprio(
        self,
        obs: dict,
        vla: Any,
        unnorm_key: str,
        num_bins: int = 256,
        state_key: str = "state",
    ) -> "ActionPromptBuilder":
        """
        Fetch + normalize + tokenize proprio → single flat string.
        """
        proprio_stats = vla.get_proprio_stats(unnorm_key=unnorm_key)
        proprio_states = obs[state_key]
        normalized = normalize_proprio(proprio=proprio_states, norm_stats=proprio_stats)
        tokenized = tokenize_proprio(normalized, num_bins=num_bins)  # list[str]
        self.proprio_tokens = "".join(tokenized)
        return self


    def build_text(self) -> str:
        """
        Render as many image blocks as we have images (main first, then wrist).
        Insert proprio tokens immediately before <thinking>.
        """
        images = self.get_images()
        image_block_count = sum(image is not None for image in images)
        image_section = IMAGE_BLOCK * image_block_count
        proprio_section = self.proprio_tokens or ""
        prompt = (
            f"{TURN_PREFIX}{image_section}"
            f"What action should the robot take to {self.task_label}?\n"
            f"Falcon:{proprio_section}{THINK_TAG}"
        )
        return prompt

    def build_inputs(self, processor) -> Tuple[dict, str]:
        """
        Returns (inputs_dict, text) so you can inspect the final prompt if needed.
        Keeps image order: [main, wrist] if both provided; omits None slots.
        """
        text = self.build_text()
        image_list = [image for image in self.get_images() if image is not None]
        inputs = processor(text=[text], images=image_list)
        return inputs, text


def invert_gripper_action_1d(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def invert_gripper_action_chunk(action):
    """
    Like normalize_gripper_action above, this will accept
      • 1-D length multiple of 7, or
      • 2-D shape (M,7).
    It flips the sign of the last element in each 7-vector.
    """
    action = np.array(action, dtype=float)
    is_action_1d = (action.ndim == 1)
    if is_action_1d:
        if action.size % ACTION_DIM != 0:
            raise ValueError(f"Length must be a multiple of {ACTION_DIM}")
        action = action.reshape(-1, ACTION_DIM)
    elif not (action.ndim == 2 and action.shape[1] == ACTION_DIM):
        raise ValueError(f"Input must be 1-D length multiple of {ACTION_DIM} or 2-D shape (M,{ACTION_DIM})")
    # flip last column
    action[:, -1] *= -1.0
    return action.ravel() if is_action_1d else action


def normalize_gripper_action_1d(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def normalize_gripper_action_chunk(action, binarize=True):
    """
    Apply to either:
      • a 1-D array of length L where L % A == 0, or
      • a 2-D array of shape (M, A).
    It reshapes 1-D→(M,A), normalizes the last column from [0,1]→[-1,1],
    optionally binarizes it to ±1, then returns the same shape as input.
    """
    action = np.array(action, dtype=float)
    is_action_1d = (action.ndim == 1)
    # reshape if needed
    if is_action_1d:
        if action.size % ACTION_DIM != 0:
            raise ValueError(f"Length must be a multiple of {ACTION_DIM}")
        action = action.reshape(-1, ACTION_DIM)
    elif not (action.ndim == 2 and action.shape[1] == ACTION_DIM):
        raise ValueError(f"Input must be 1-D length multiple of {ACTION_DIM} or 2-D shape (M,{ACTION_DIM})")
    # normalize last column
    orig_low, orig_high = 0.0, 1.0
    col = action[:, -1]
    normed = 2 * (col - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        normed = np.sign(normed)
    action[:, -1] = normed
    # restore original shape
    return action.ravel() if is_action_1d else action


def normalize_gripper_action(action: np.array, binarize=True):
    """This function is a router for normalizing gripper actions with and without time horizon"""
    # No Action Chunking
    if len(action.shape) == 1 and action.shape[0] == ACTION_DIM:
        return normalize_gripper_action_1d(action, binarize)
    # Action Chunking (condition for taking care of H*A tokens or (H, A) tokens)
    elif len(action.shape) == 2 or ((len(action.shape) == 1) and (action.shape[0] % ACTION_DIM == 0)):
        return normalize_gripper_action_chunk(action, binarize)
    else:
        raise Exception(f'Action Shape of size {len(action.shape)} is not supported')


def invert_gripper_action(action: np.array):
    """This function is a router for inverting gripper actions with and without time horizon"""
    # No Action Chunking
    if len(action.shape) == 1 and action.shape[0] == ACTION_DIM:
        return invert_gripper_action_1d(action)
    # Action Chunking (condition for taking care of (H, A) tokens or (H*A) tokens)
    elif len(action.shape) == 2 or (len(action.shape) == 1 and action.shape[0] % ACTION_DIM == 0):
        return invert_gripper_action_chunk(action)
    else:
        raise Exception(f'Action Shape of size {len(action.shape)} is not supported')


def load_falcon_model(model_name: str, hf_token: Optional[str]) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).to(DEVICE)
    return processor, model


@torch.no_grad()
def predict_actions_chunk(
    vla,
    processor,
    observation,         # dict[str, np.ndarray (H,W,3) uint8]
    task_label: str,
    horizon: int = 5,
    unnorm_key: str = "ee_delta",
    use_wrist: bool = False,
    use_secondary: bool = False,
    use_proprio: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
        actions: (H, D) tensor on DEVICE
    """
    image = observation["primary_image"]
    cv2.imwrite(f"debug/images/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", image)
    input_builder = ActionPromptBuilder(task_label=task_label).add_main_image(image=image)
    if use_wrist:
        input_builder.add_wrist_image(observation["wrist_image"])
    if use_secondary:
        input_builder.add_secondary_image(observation["secondary_image"])

    if use_proprio:
        input_builder.add_proprio(
            obs=observation,
            vla=vla,
            unnorm_key=unnorm_key,
            num_bins=256,
            state_key="state",
        )

    inputs, _ = input_builder.build_inputs(processor=processor)
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    actions = vla.predict_action(inputs, unnorm_key=unnorm_key, horizon=horizon, do_sample=False)
    return actions
