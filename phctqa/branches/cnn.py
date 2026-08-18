# phctqa/branches/cnn.py
"""CNN classifier adapter for PHCTQA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBranchAdapter
from phctqa.io import load_volume
from phctqa.preprocess import build_cnn_preprocess, Seg2Zero, MIP

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def _replace_first_conv(model: nn.Module, backbone: str, in_channels: int) -> None:
    if in_channels == 3:
        return

    if backbone.startswith("efficientnet"):
        parent = model.features[0]
        first_conv = parent[0]
        attr_path = [0, 0]
    elif backbone.startswith("resnet"):
        first_conv = model.conv1
        attr_path = ["conv1"]
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        else:
            raise RuntimeError(f"Cannot find first Conv2d for {backbone}")
        attr_path = None

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels <= 3:
            new_conv.weight[:, :in_channels, :, :] = first_conv.weight[
                :, :in_channels, :, :
            ]
        else:
            new_conv.weight[:, :3, :, :] = first_conv.weight
            for c in range(3, in_channels):
                new_conv.weight[:, c, :, :] = first_conv.weight[
                    :, (c - 3) % 3, :, :
                ]

    if attr_path is not None:
        obj = model
        for part in attr_path[:-1]:
            obj = getattr(obj, str(part)) if isinstance(part, str) else obj[part]
        last = attr_path[-1]
        if isinstance(last, str):
            setattr(obj, last, new_conv)
        else:
            obj[last] = new_conv
    else:
        raise RuntimeError("Manual first-conv replacement unsupported for this backbone")

    logger.info(
        f"[CNNAdapter] Modified first conv: in_channels 3 -> {in_channels} "
        f"for {backbone}"
    )


def build_model(
    backbone: str,
    num_classes: int,
    dropout: float = 0.0,
    in_channels: int = 3,
) -> nn.Module:
    if backbone == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s

        model = efficientnet_v2_s(weights=None)
        in_features = model.classifier[-1].in_features
        if dropout > 0:
            model.classifier = nn.Sequential(
                model.classifier[0],
                nn.Sequential(
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Linear(in_features, num_classes),
                ),
            )
        else:
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if in_channels != 3:
        _replace_first_conv(model, backbone, in_channels)

    return model


# --------------------------------------------------------------------------- #
# View extractors (task-oriented multi-view integration)
# --------------------------------------------------------------------------- #

def extract_mid_sagittal(volume: np.ndarray) -> np.ndarray:
    """Extract the centre sagittal slice from a [Z, Y, X] volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")
    x_center = volume.shape[2] // 2
    return volume[:, :, x_center]


def extract_axial_slice(volume: np.ndarray, position: str = "bottom") -> np.ndarray:
    """Extract the uppermost or lowermost axial slice from a [Z, Y, X] volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")
    if position == "bottom":
        return volume[-1, :, :]
    elif position == "top":
        return volume[0, :, :]
    else:
        raise ValueError(f"position must be 'top' or 'bottom', got {position}")


def extract_bone_mip(
    volume: np.ndarray,
    hu_min: float,
    hu_max: float,
    threshold: float,
) -> np.ndarray:
    """Bone-segmented axial MIP with 0-255 normalization."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")

    seg = Seg2Zero(hu_min=hu_min, hu_max=hu_max, threshold=threshold, target="higher")
    bone_3d = seg(volume)

    mip = MIP(axis=0)
    img_2d = mip(bone_3d)

    return img_2d.astype(np.float32)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

class CNNAdapter(BaseBranchAdapter):
    def load(self) -> None:
        cnn_cfg = self.branch_cfg.get("cnn", {})

        def _get(key: str, default: Any) -> Any:
            if key in cnn_cfg:
                return cnn_cfg[key]
            if key in self.train_cfg:
                return self.train_cfg[key]
            return default

        self.view = str(_get("view", "mid_sagittal"))
        self.backbone = str(_get("backbone", "efficientnet_v2_s"))
        self.num_classes = int(_get("num_classes", 2))
        self.input_size = int(_get("input_size", 224))
        self.in_channels = int(_get("in_channels", 3))
        self.dropout = float(_get("dropout", 0.0))

        pre_cfg = cnn_cfg.get("preprocess", {})
        self.window_width = float(
            pre_cfg.get("window_width", _get("window_width", 100))
        )
        self.window_center = float(
            pre_cfg.get("window_center", _get("window_center", 50))
        )
        cc = pre_cfg.get("center_crop", _get("center_crop", None))
        self.center_crop_size = int(cc) if cc is not None else None

        bone_cfg = cnn_cfg.get("bone_segmentation", {})
        self.bone_hu_min = float(bone_cfg.get("hu_min", _get("hu_min", 0)))
        self.bone_hu_max = float(bone_cfg.get("hu_max", _get("hu_max", 3500)))
        self.bone_threshold = float(
            bone_cfg.get("threshold", _get("threshold", 200))
        )

        # Build and cache preprocessing transform
        if self.view == "mid_sagittal":
            self._transform = build_cnn_preprocess(
                input_size=self.input_size,
                window_width=self.window_width,
                window_center=self.window_center,
                center_crop=None,
                repeat_channels=True,
            )
        elif self.view == "bone_mip":
            self._transform = build_cnn_preprocess(
                input_size=self.input_size,
                center_crop=self.center_crop_size,
                repeat_channels=True,
            )
        elif self.view in ("axial_bottom", "axial_top"):
            self._transform = build_cnn_preprocess(
                input_size=self.input_size,
                window_width=self.window_width,
                window_center=self.window_center,
                center_crop=self.center_crop_size,
                repeat_channels=True,
            )
        else:
            raise ValueError(f"Unsupported view: {self.view}")

        # Load model
        ckpt_path = self.require_weights()
        self.model = build_model(
            backbone=self.backbone,
            num_classes=self.num_classes,
            dropout=self.dropout,
            in_channels=self.in_channels,
        )
        self._load_checkpoint(self.model, ckpt_path)
        self.model.eval().to(self.device)

        logger.info(
            f"{self.name}: CNNAdapter ready "
            f"(view={self.view}, backbone={self.backbone}, device={self.device})"
        )

    def _load_checkpoint(self, model: nn.Module, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = (
            ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt
        )
        cleaned = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)

    def predict(self, input_path: str) -> Dict[str, Any]:
        volume, meta = load_volume(input_path)
        volume_np = self._to_numpy(volume)

        # Extract task-specific 2D view
        if self.view == "mid_sagittal":
            img_2d = extract_mid_sagittal(volume_np)
            slice_meta = {"slice_axis": "sagittal", "slice_side": "centre"}
        elif self.view == "bone_mip":
            img_2d = extract_bone_mip(
                volume_np,
                hu_min=self.bone_hu_min,
                hu_max=self.bone_hu_max,
                threshold=self.bone_threshold,
            )
            slice_meta = {"slice_axis": "axial", "slice_side": "mip"}
        elif self.view == "axial_bottom":
            img_2d = extract_axial_slice(volume_np, position="bottom")
            slice_meta = {"slice_axis": "axial", "slice_side": "bottom"}
        elif self.view == "axial_top":
            img_2d = extract_axial_slice(volume_np, position="top")
            slice_meta = {"slice_axis": "axial", "slice_side": "top"}
        else:
            raise ValueError(f"Unsupported view: {self.view}")

        # Apply cached albumentations pipeline
        result = self._transform(image=img_2d)
        tensor = torch.from_numpy(result["image"]).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        prob_positive = float(probs[0, 1].cpu().numpy())

        result = self.result(prob_positive, self.threshold)
        result["details"] = {
            "view": self.view,
            "input_size": self.input_size,
            "backbone": self.backbone,
        }
        result["meta"] = slice_meta
        return result

    @staticmethod
    def _to_numpy(volume: Any) -> np.ndarray:
        if hasattr(volume, "GetArrayFromImage"):
            import SimpleITK as sitk
            return sitk.GetArrayFromImage(volume)
        if hasattr(volume, "numpy"):
            return volume.numpy()
        arr = np.asarray(volume)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3-D volume, got shape {arr.shape}")
        return arr