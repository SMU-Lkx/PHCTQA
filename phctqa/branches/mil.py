# phctqa/adapters/mil.py
"""Attention-based MIL adapter for PHCTQA.

Implements the **physics-informed multiple instance learning** (PIMIL)
architecture described in the manuscript.

Supported branches:
  - head_MOT  (brain_foreground_axi + foreground_screening)
  - thorax_MOT (otsu_lung_axi + otsu_lung)
  - thorax_MTL (metal_streak_axi + intensity_threshold_cc, with metal pre-screening)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBranchAdapter
from phctqa.io import load_volume
from phctqa.preprocess import build_cnn_preprocess, foreground_ratio

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backbone feature extractor
# --------------------------------------------------------------------------- #

def build_backbone(backbone: str, dropout: float = 0.0) -> nn.Module:
    """Build a feature extractor from torchvision (classification head removed)."""
    if backbone == "efficientnet_v2_s":
        from torchvision.models import efficientnet_v2_s

        model = efficientnet_v2_s(weights=None)
        model.classifier = nn.Identity()
    elif backbone == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def get_feature_dim(backbone: str) -> int:
    if backbone == "efficientnet_v2_s":
        return 1280
    elif backbone == "resnet50":
        return 2048
    raise ValueError(f"Unknown feature dim for {backbone}")


# --------------------------------------------------------------------------- #
# Gated attention mechanism
# --------------------------------------------------------------------------- #

class MILModel(nn.Module):
    """Single-bag MIL model.  Architecture must stay in sync with
    Models/MotArtDet_Breath/ABMIL.py::GatedAttentionMIL.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        hidden_dim: int = 256,
        drop_inst: float = 0.3,
        drop_att: float = 0.2,
        in_channels: int = 3,
    ):
        super().__init__()
        raw_backbone = build_backbone(backbone, dropout=0.0)

        # instance_encoder = Sequential(features, pool, flatten)
        # For EfficientNet-V2-S: children() -> [features, avgpool, classifier]
        # We drop the final classifier and keep features+avgpool.
        children = list(raw_backbone.children())[:-1]
        if len(children) > 0 and isinstance(
            children[-1], (nn.AdaptiveAvgPool2d, nn.AvgPool2d)
        ):
            self.instance_encoder = nn.Sequential(*children, nn.Flatten())
        else:
            # Fallback (should not happen for standard backbones)
            self.instance_encoder = nn.Sequential(
                *children,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        self.feature_dim = get_feature_dim(backbone)
        self.drop_inst = nn.Dropout(drop_inst)
        self.V = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.U = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.drop_att = nn.Dropout(drop_att)
        self.w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes),
        )

        if in_channels != 3:
            self._patch_first_conv(in_channels, raw_backbone)

    def _patch_first_conv(self, in_channels: int, raw_backbone: nn.Module) -> None:
        first_conv = None
        for m in raw_backbone.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        if first_conv is None or first_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        with torch.no_grad():
            orig = first_conv.weight.data
            new_weight = orig.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            if in_channels <= 3:
                new_weight[:, :in_channels] = orig[:, :in_channels]
            new_conv.weight.copy_(new_weight)
            if new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias.data)

        if hasattr(raw_backbone, "features"):
            feat0 = raw_backbone.features[0]
            if isinstance(feat0, nn.Sequential) and hasattr(feat0, "0"):
                feat0[0] = new_conv
            else:
                raw_backbone.features[0] = new_conv
        elif hasattr(raw_backbone, "conv1"):
            raw_backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, C, H, W]  (single bag, B=1)
        h = self.instance_encoder(x)  # [M, feature_dim]
        if h.dim() > 2:
            h = h.view(h.size(0), -1)

        h = self.drop_inst(h)
        A_V = self.V(h)                 # [M, hidden_dim]
        A_U = self.U(h)                 # [M, hidden_dim]
        x = self.drop_att(A_V * A_U)    # [M, hidden_dim]

        # Bypass oneDNN matmul primitive bug (triggered by torch_npu in this env).
        # self.w is nn.Linear(hidden_dim, 1); expanded into element-wise ops.
        A = (x * self.w.weight).sum(dim=-1) + self.w.bias  # [M]

        A = torch.softmax(A, dim=0)

        bag = (A.unsqueeze(-1) * h).sum(dim=0, keepdim=True)  # [1, feature_dim]
        logits = self.classifier(bag)
        return logits


# --------------------------------------------------------------------------- #
# Physics-informed slice selection
# --------------------------------------------------------------------------- #

def select_slices_foreground(
    volume: np.ndarray,
    min_ratio: float = 0.05,
    air_thresh: int = -100,
) -> List[int]:
    """Select slices with sufficient foreground."""
    indices = []
    for i in range(volume.shape[0]):
        ratio = foreground_ratio(volume[i], air_thresh)
        if ratio >= min_ratio:
            indices.append(i)
    return indices


def select_slices_otsu_lung(
    volume: np.ndarray,
    min_area: int = 100,
    min_regions: int = 1,
) -> List[int]:
    """Otsu thresholding + connected-component analysis for lung parenchyma."""
    from skimage.filters import threshold_otsu
    from skimage import measure

    indices = []
    for i in range(volume.shape[0]):
        img = volume[i]
        try:
            thresh = threshold_otsu(img)
        except ValueError:
            continue
        binary = img > thresh
        regions = measure.label(binary, connectivity=1)
        h, w = binary.shape
        valid = 0
        for region in measure.regionprops(regions):
            if region.area < min_area:
                continue
            minr, minc, maxr, maxc = region.bbox
            touches_border = minr == 0 or minc == 0 or maxr >= h - 1 or maxc >= w - 1
            if not touches_border:
                valid += 1
        if valid >= min_regions:
            indices.append(i)
    return indices


def select_slices_intensity_cc(
    volume: np.ndarray,
    intensity_threshold: float = 2000,
    min_area: int = 10,
) -> List[int]:
    """Intensity threshold + connected-component analysis for metal streaks."""
    from skimage import measure

    indices = []
    for i in range(volume.shape[0]):
        img = volume[i]
        binary = img >= intensity_threshold
        if not np.any(binary):
            continue
        regions = measure.label(binary, connectivity=1)
        max_area = max((r.area for r in measure.regionprops(regions)), default=0)
        if max_area >= min_area:
            indices.append(i)
    return indices


# --------------------------------------------------------------------------- #
# MIL Adapter
# --------------------------------------------------------------------------- #

class MILAdapter(BaseBranchAdapter):
    """MIL branch adapter for sparse, heterogeneous defects (motion / metal)."""

    def load(self) -> None:
        mil_cfg = self.branch_cfg.get("mil", {})

        def _get(key: str, default: Any) -> Any:
            if key in mil_cfg:
                return mil_cfg[key]
            # Safe fallback: train_cfg may not exist on BaseBranchAdapter
            train_cfg = getattr(self, "train_cfg", None)
            if train_cfg is not None and key in train_cfg:
                return train_cfg[key]
            return default

        self.bag_type = str(_get("bag", "brain_foreground_axi"))
        self.prior = str(_get("prior", "foreground_screening"))
        self.backbone = str(_get("backbone", "efficientnet_v2_s"))
        self.num_classes = int(_get("num_classes", 2))
        self.input_size = int(_get("input_size", 224))
        self.in_channels = int(_get("in_channels", 3))
        self.dropout = float(_get("dropout", 0.0))
        self.hidden_dim = int(_get("hidden_dim", 256))

        pre_cfg = mil_cfg.get("preprocess", {})
        self.window_width = float(
            pre_cfg.get("window_width", _get("window_width", 100))
        )
        self.window_center = float(
            pre_cfg.get("window_center", _get("window_center", 50))
        )
        cc = pre_cfg.get("center_crop", _get("center_crop", None))
        self.center_crop_size = int(cc) if cc is not None else None

        self.prior_params = self._load_prior_params(mil_cfg)

        # Build and cache albumentations pipeline (exactly matches training)
        self._transform = build_cnn_preprocess(
            input_size=self.input_size,
            window_width=self.window_width,
            window_center=self.window_center,
            center_crop=self.center_crop_size,
            repeat_channels=True,
        )

        ckpt_path = self.require_weights()
        self.model = MILModel(
            backbone=self.backbone,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            in_channels=self.in_channels,
        )
        self._load_checkpoint(self.model, ckpt_path)
        self.model.eval().to(self.device)

        logger.info(
            f"{self.name}: MILAdapter ready "
            f"(prior={self.prior}, backbone={self.backbone}, device={self.device})"
        )

    def _load_prior_params(self, mil_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Load physics-informed prior parameters."""
        if self.prior == "foreground_screening":
            cfg = mil_cfg.get("foreground_screening", {})
            return {
                "min_ratio": float(cfg.get("min_ratio", 0.05)),
                "air_thresh": int(cfg.get("air_thresh", -100)),
            }
        elif self.prior == "otsu_lung":
            cfg = mil_cfg.get("otsu_lung", {})
            return {
                "min_area": int(cfg.get("min_area", 100)),
                "min_regions": int(cfg.get("min_regions", 1)),
            }
        elif self.prior == "intensity_threshold_cc":
            cfg = mil_cfg.get("intensity_threshold_cc", {})
            prescreen_cfg = cfg.get("metal_prescreen", {})
            return {
                "intensity_threshold": float(
                    cfg.get("intensity_threshold", 2000)
                ),
                "min_area": int(cfg.get("min_area", 10)),
                "metal_prescreen": bool(prescreen_cfg.get("enabled", True)),
                "metal_thresh": int(prescreen_cfg.get("metal_thresh", 2000)),
                "min_metal_pixels": int(
                    prescreen_cfg.get("min_metal_pixels", 10)
                ),
            }
        else:
            raise ValueError(f"Unknown prior: {self.prior}")

    def _load_checkpoint(self, model: nn.Module, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = (
            ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        )
        # Strip both 'module.' (DataParallel) and 'model.' (common wrapper prefix)
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ("module.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=True)

    def predict(self, input_path: str) -> Dict[str, Any]:
        volume, meta = load_volume(input_path)
        volume_np = self._to_numpy(volume)  # [Z, Y, X]

        # ------------------------------------------------------------------
        # Metal pre-screening (thorax_MTL only)
        # ------------------------------------------------------------------
        if self.prior == "intensity_threshold_cc":
            params = self.prior_params
            if params.get("metal_prescreen", True):
                global_metal = np.sum(volume_np > params["metal_thresh"])
                if global_metal < params["min_metal_pixels"]:
                    logger.info(
                        f"{self.name}: no metal detected ({global_metal} pixels < "
                        f"{params['min_metal_pixels']}), returning prelabeled negative"
                    )
                    result = self.result(0.0, self.threshold)
                    result["details"] = {
                        "prelabeled": True,
                        "reason": "no_metal",
                        "metal_pixels": int(global_metal),
                        "metal_thresh": params["metal_thresh"],
                    }
                    result["meta"] = {
                        "view": "axial",
                        "slice_selection": self.prior,
                    }
                    return result

        # 1. Physics-informed slice selection
        selected_indices = self._select_slices(volume_np)
        if not selected_indices:
            logger.warning(
                f"{self.name}: no slices passed prior screening; "
                "returning default (negative)"
            )
            result = self.result(0.0, self.threshold)
            result["details"] = {
                "bag_size": 0,
                "reason": "no_slices_passed_prior",
                "prior": self.prior,
            }
            result["meta"] = {
                "view": "axial",
                "slice_selection": self.prior,
            }
            return result

        # 2. Build bag tensor from selected slices
        bag_tensor = self._build_bag(volume_np, selected_indices)  # [M, C, H, W]

        # 3. MIL inference
        with torch.no_grad():
            logits = self.model(bag_tensor)  # [1, num_classes]
            probs = F.softmax(logits, dim=1)

        prob_positive = float(probs[0, 1].cpu().numpy())

        result = self.result(prob_positive, self.threshold)
        result["details"] = {
            "bag_size": len(selected_indices),
            "selected_indices": selected_indices,
            "prior": self.prior,
            "backbone": self.backbone,
        }
        result["meta"] = {
            "view": "axial",
            "slice_selection": self.prior,
        }
        return result

    def _select_slices(self, volume: np.ndarray) -> List[int]:
        """Apply physics-informed slice selection prior."""
        if self.prior == "foreground_screening":
            return select_slices_foreground(
                volume,
                min_ratio=self.prior_params["min_ratio"],
                air_thresh=self.prior_params["air_thresh"],
            )
        elif self.prior == "otsu_lung":
            return select_slices_otsu_lung(
                volume,
                min_area=self.prior_params["min_area"],
                min_regions=self.prior_params["min_regions"],
            )
        elif self.prior == "intensity_threshold_cc":
            return select_slices_intensity_cc(
                volume,
                intensity_threshold=self.prior_params["intensity_threshold"],
                min_area=self.prior_params["min_area"],
            )
        else:
            raise ValueError(f"Unknown prior: {self.prior}")

    def _build_bag(self, volume: np.ndarray, indices: List[int]) -> torch.Tensor:
        """Extract and preprocess selected slices into a bag tensor."""
        slices = []
        for idx in indices:
            img = volume[idx]  # [H, W] raw HU
            # Apply cached albumentations pipeline (pixel-level match with training)
            result = self._transform(image=img)
            tensor = torch.from_numpy(result["image"])  # [C, H, W]
            slices.append(tensor)
        bag = torch.stack(slices, dim=0)  # [M, C, H, W]
        return bag.to(self.device)

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