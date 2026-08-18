# phctqa/adapters/tool.py
"""Rule-based (traditional image processing) adapter for PHCTQA.

Replaces the legacy tool.py; renamed to ``tool.py`` to align with the
manuscript terminology (Tool ①–⑩ in Fig. 2A).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu

from .base import BaseBranchAdapter
from phctqa.io import load_volume, uppermost_axial_slice
from phctqa.preprocess import build_tool_preprocess


@dataclass(frozen=True)
class RuleParams:
    criterion: str                 # "nonzero_ratio" | "otsu_internal_regions"
    crop_size: int
    window_width: float
    window_center: float
    nonzero_ratio: Optional[float] = None
    min_area: Optional[int] = None
    min_regions: Optional[int] = None


class ToolAdapter(BaseBranchAdapter):
    """Traditional image-processing branch (e.g. incomplete vertex/apex scan)."""

    def load(self) -> None:
        r = self.branch_cfg["tool"]
        self.params = RuleParams(
            criterion=r["criterion"],
            crop_size=int(r["crop_size"]),
            window_width=float(r["window_width"]),
            window_center=float(r["window_center"]),
            nonzero_ratio=r.get("nonzero_ratio"),
            min_area=r.get("min_area"),
            min_regions=r.get("min_regions"),
        )

        if self.params.criterion == "nonzero_ratio" and self.params.nonzero_ratio is None:
            raise ValueError(f"{self.name}: nonzero_ratio is required for criterion=nonzero_ratio")
        if self.params.criterion == "otsu_internal_regions":
            if self.params.min_area is None or self.params.min_regions is None:
                raise ValueError(f"{self.name}: min_area and min_regions are required for otsu_internal_regions")

        # Cache the albumentations pipeline (pixel-level match with training)
        self._transform = build_tool_preprocess(
            window_width=self.params.window_width,
            window_center=self.params.window_center,
            crop_size=self.params.crop_size,
        )

    def _prepare_uppermost_slice(self, input_path: str) -> np.ndarray:
        volume, meta = load_volume(input_path)
        img = uppermost_axial_slice(volume)  # 2-D numpy [H, W]
        result = self._transform(image=img)
        return result["image"]  # uint8 [0, 255]

    def predict(self, input_path: str) -> Dict[str, Any]:
        img = self._prepare_uppermost_slice(input_path)

        if self.params.criterion == "nonzero_ratio":
            ratio = float(np.count_nonzero(img) / img.size)
            pred = int(ratio > float(self.params.nonzero_ratio))
            details = {
                "nonzero_ratio": ratio,
                "tool_threshold": float(self.params.nonzero_ratio),
                "pixel_space": "windowed_uint8",
            }
        elif self.params.criterion == "otsu_internal_regions":
            thresh = threshold_otsu(img)
            binary = img > thresh
            regions = measure.label(~binary, connectivity=1)
            h, w = binary.shape
            internal = 0
            for region in measure.regionprops(regions):
                if region.area < int(self.params.min_area):
                    continue
                minr, minc, maxr, maxc = region.bbox
                touches_border = minr == 0 or minc == 0 or maxr >= h - 1 or maxc >= w - 1
                if not touches_border:
                    internal += 1
            pred = int(internal >= int(self.params.min_regions))
            details = {
                "internal_regions": internal,
                "min_area": int(self.params.min_area),
                "min_regions": int(self.params.min_regions),
                "pixel_space": "windowed_uint8",
            }
        else:
            raise ValueError(f"unknown tool criterion: {self.params.criterion}")

        return {
            "pred": pred,
            "pred_meaning": "1 = defect present, 0 = absent",
            "probability": None,
            "threshold": None,
            "status": "ok",
            "details": details,
            "meta": {"slice_axis": "axial", "slice_side": "uppermost"},
        }