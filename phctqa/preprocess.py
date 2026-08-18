# PHCTQA/phctqa/preprocess.py
"""Preprocessing and view-extraction utilities for PHCTQA.

All branches (rule/tool, CNN, MIL) share albumentations-based transforms
originally defined in ``data/preprocess/Transforms.py`` to guarantee
pixel-level consistency between training and inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


# --------------------------------------------------------------------------- #
# Albumentations custom transforms
# --------------------------------------------------------------------------- #

class AdjustWindow_A(ImageOnlyTransform):
    """
    HU windowing → uint8 [0, 255].
    Reproduces the training-time transform used in all CNN/MIL/toolpipelines.
    """

    def __init__(self, window_width, window_center, always_apply=True):
        super().__init__(always_apply)
        self.window_width = window_width
        self.window_center = window_center

    def apply(self, image, **params):
        min_value = self.window_center - (self.window_width / 2)
        max_value = self.window_center + (self.window_width / 2)
        image = image.astype(np.float32)
        adjusted = np.clip(image, min_value, max_value)
        adjusted = ((adjusted - min_value) / self.window_width) * 255
        return adjusted.astype(np.uint8)

    def get_transform_init_args_names(self):
        return "window_width", "window_center"


def fix_data_format(x, **kwargs):
    """Squeeze and cast to float32 (handles PIL or numpy input)."""
    return np.array(x).squeeze().astype(np.float32)


def ensure_channel_dim(x, **kwargs):
    """Ensure trailing channel dim: (H, W) -> (H, W, 1)."""
    return x if x.ndim == 3 else np.expand_dims(x, -1)


def to_chw(x, **kwargs):
    """HWC -> CHW."""
    return np.transpose(x, (2, 0, 1))


def repeat_3ch(img, **kwargs):
    """Repeat single-channel CHW to 3-channel CHW."""
    if img.shape[0] == 1:
        return np.repeat(img, 3, axis=0)
    return img


def foreground_ratio(hu_slice: np.ndarray, air_thresh: int = -100) -> float:
    """Compute foreground ratio (non-air pixel proportion)."""
    fg_mask = hu_slice > air_thresh
    return float(fg_mask.sum() / fg_mask.size)

# --------------------------------------------------------------------------- #
# Physics-informed view extraction
# --------------------------------------------------------------------------- #

class Seg2Zero:
    """Segmentation-by-threshold + normalization to [0, 255].

    Replicates training-time ``BoneMIP_A`` preprocessing:
      1. mask by threshold
      2. clip to [hu_min, hu_max]
      3. linear map to [0, 255]
    """

    def __init__(self, hu_min, hu_max, threshold, target="higher"):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.threshold = threshold
        self.target = target

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        if self.target == "higher":
            mask = image_array >= self.threshold
        else:
            mask = image_array <= self.threshold
        processed = image_array * mask
        windowed = np.clip(processed, self.hu_min, self.hu_max)
        normalized = (windowed - self.hu_min) / (self.hu_max - self.hu_min + 1e-6) * 255
        return normalized


class MIP:
    """Maximum intensity projection along a given axis."""

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Input must be a 3-D NumPy array")
        return np.max(image, axis=self.axis)


# --------------------------------------------------------------------------- #
# Pipeline builders
# --------------------------------------------------------------------------- #

def build_tool_preprocess(
    window_width: float,
    window_center: float,
    crop_size: int,
) -> A.Compose:
    """Build the albumentations preprocessing pipeline for tool branches.

    Applies windowing → center crop, producing a uint8 [0, 255] 2-D slice
    ready for traditional image analysis (nonzero_ratio, Otsu, etc.).
    """
    return A.Compose([
        AdjustWindow_A(window_width, window_center),
        A.CenterCrop(height=crop_size, width=crop_size),
    ])


def build_cnn_preprocess(
    input_size: int = 224,
    center_crop: Optional[int] = None,
    window_width: Optional[float] = None,
    window_center: Optional[float] = None,
    repeat_channels: bool = True,
) -> A.Compose:
    """Build the canonical albumentations preprocessing pipeline for CNN branches.

    Exactly reproduces the training-time ``A.Compose`` used in
    ``infer_CNN.py`` and ``train_CNN.py``.
    """
    transforms = []

    if window_width is not None and window_center is not None:
        transforms.append(AdjustWindow_A(window_width, window_center))

    transforms.append(A.Lambda(image=fix_data_format, name="fix_data_format"))
    transforms.append(A.Lambda(image=ensure_channel_dim, name="ensure_channel_dim"))

    if center_crop is not None and center_crop > 0:
        transforms.append(A.CenterCrop(height=center_crop, width=center_crop))

    transforms.append(
        A.Resize(height=input_size, width=input_size, interpolation=1)
    )
    transforms.append(
        A.Normalize(
            mean=[0.5 * 255],
            std=[0.5 * 255],
            max_pixel_value=255.0,
        )
    )
    transforms.append(A.Lambda(image=to_chw, name="to_chw"))

    if repeat_channels:
        transforms.append(A.Lambda(image=repeat_3ch, name="repeat_3ch"))

    return A.Compose(transforms)
