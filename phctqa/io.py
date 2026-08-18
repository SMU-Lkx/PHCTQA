# phctqa/io.py
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk


def _technical_metadata(dicom_file: str) -> Dict[str, Any]:
    ds = pydicom.dcmread(dicom_file)
    return {
        "Modality": ds.get("Modality", "N/A"),
        "BodyPartExamined": ds.get("BodyPartExamined", "N/A"),
        "ImageType": ds.get("ImageType", "N/A"),
        "Manufacturer": ds.get("Manufacturer", "N/A"),
        "ManufacturerModelName": ds.get("ManufacturerModelName", "N/A"),
        "RescaleSlope": float(ds.get("RescaleSlope", 1.0)),
        "RescaleIntercept": float(ds.get("RescaleIntercept", 0.0)),
        "WindowCenter": ds.get("WindowCenter", None),
        "WindowWidth": ds.get("WindowWidth", None),
    }


def load_dicom_series(folder_path: str) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Load one DICOM series and enforce the PHCTQA convention:
        numpy volume[0] == uppermost/head-side axial slice.

    No silent fallback: any read/orientation failure raises.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"expect a DICOM directory, got: {folder}")

    reader = sitk.ImageSeriesReader()
    reader.SetNumberOfThreads(1)
    dicom_names = reader.GetGDCMSeriesFileNames(str(folder))
    if not dicom_names:
        raise FileNotFoundError(f"no DICOM series found in {folder}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    meta = _technical_metadata(dicom_names[0])

    size = image.GetSize()
    if size[2] < 2:
        raise ValueError(f"need at least 2 axial slices to determine direction, got size={size}")

    pos0 = image.TransformIndexToPhysicalPoint((0, 0, 0))
    pos1 = image.TransformIndexToPhysicalPoint((0, 0, size[2] - 1))
    idx0_is_head = pos0[2] > pos1[2]
    meta["slice_direction"] = "idx0=head" if idx0_is_head else "idx0=foot"

    if not idx0_is_head:
        orig_dir = np.array(image.GetDirection()).reshape(3, 3)
        image.SetDirection(orig_dir.ravel())
        image = sitk.Flip(image, flipAxes=(False, False, True))
        new_dir = orig_dir.copy()
        new_dir[:, 2] = -new_dir[:, 2]
        image.SetDirection(new_dir.ravel())
        meta["slice_direction"] = "flipped_to_idx0=head"

    return image, meta


def load_volume(input_path: str):
    image, meta = load_dicom_series(input_path)
    volume = sitk.GetArrayFromImage(image)  # [z, y, x]
    if volume.ndim != 3:
        raise ValueError(f"expect 3D volume [z,y,x], got {volume.shape}")
    meta["shape_zyx"] = tuple(int(x) for x in volume.shape)
    meta["spacing_xyz"] = tuple(float(x) for x in image.GetSpacing())
    return volume, meta


def uppermost_axial_slice(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"expect volume [z,y,x], got shape {volume.shape}")
    return np.asarray(volume[0])
