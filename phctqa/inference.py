#!/usr/bin/env python3
"""Config-driven unified inference for PHCTQA.

Usage:
    python inference.py --config config/inference/head.yaml --input example/head_dicom --output output/report.json
    python inference.py --config config/inference/thorax.yaml --input example/thorax_dicom --output output/report.json
"""
import argparse, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# PHCTQA/
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(p):
    if p is None:
        return None
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


from phctqa.util.infer_util import load_yaml
from phctqa.registry import BRANCH_TYPES, BranchSpec, import_adapter
from functools import lru_cache
import phctqa.io

_original_load_volume = phctqa.io.load_volume

@lru_cache(maxsize=2)
def _cached_load_volume(path: str):
    return _original_load_volume(path)

phctqa.io.load_volume = _cached_load_volume


def _predict_branch(adapter, input_path: str):
    t0 = time.time()
    result = adapter.predict(input_path)
    result["_branch_elapsed_sec"] = round(time.time() - t0, 3)
    return result

def main() -> int:
    ap = argparse.ArgumentParser(description="PHCTQA unified single-case inference")
    ap.add_argument(
        "--config",
        default="config/inference/head.yaml",
        help="Inference config path (relative to repo root or absolute)",
    )
    ap.add_argument(
        "--input",
        default=None,
        help="Input DICOM directory or NIfTI file",
    )
    ap.add_argument(
        "--weights-dir",
        default="weight",
        help="Directory containing model weights",
    )
    ap.add_argument(
        "--output",
        default="output/report.json",
        help="Output JSON report path",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "npu"],
        default="cpu",
        help="Inference device",
    )
    args = ap.parse_args()

    t0 = time.time()
    config_path = resolve_repo_path(args.config)
    cfg = load_yaml(config_path)

    region = str(cfg.get("region", "head")).lower()
    if region not in {"head", "thorax"}:
        raise ValueError(f"region must be head/thorax, got {region!r}")

    device = args.device or cfg.get("device", "cpu")

    input_path = args.input or (cfg.get("input") or {}).get("path")
    if not input_path:
        raise ValueError(
            "No input path. Set --input or input.path in config."
        )
    input_path = str(resolve_repo_path(input_path))

    weights_dir = resolve_repo_path(args.weights_dir or cfg.get("weights_dir", "weights"))
    output_path = resolve_repo_path(args.output)

    report = {
        "system": "PHCTQA",
        "config": str(config_path),
        "region": region,
        "device": device,
        "input": input_path,
        "status": "ok",
        "defects": {},
        "elapsed_sec": None,
    }

    # ------------------------------------------------------------------
    # 1. Load all models sequentially (to prevent IO or VRAM conflicts
    #    caused by concurrent weight loading across multiple processes/threads)
    # ------------------------------------------------------------------
    adapters = []
    for name, bcfg in (cfg.get("branches") or {}).items():
        if not bcfg.get("enabled", True):
            continue
        spec = BranchSpec(
            name=name,
            type=bcfg["type"],
            weights=bcfg.get("weight"),
            enabled=True,
        )
        adapter_cls = import_adapter(BRANCH_TYPES[spec.type])
        adapter = adapter_cls(
            name=name,
            branch_cfg={
                **bcfg,
                "weights_path": str(weights_dir / spec.weights) if spec.weights else None,
            },
            global_cfg=cfg,
        )
        adapter.load()
        adapters.append((name, adapter))

    # ------------------------------------------------------------------
    # 2. Parallel inference (ThreadPoolExecutor: PyTorch/NumPy release
    #    the GIL at the backend, enabling true multi-core parallelism
    #    on CPU; GPU/NPU can also run in parallel if VRAM is sufficient)
    # ------------------------------------------------------------------
    t_infer = time.time()
    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        futures = {
            executor.submit(_predict_branch, adapter, input_path): name
            for name, adapter in adapters
        }
        for future in as_completed(futures):
            name = futures[future]
            report["defects"][name] = future.result()

    report["elapsed_sec"] = round(time.time() - t0, 3)
    report["inference_sec"] = round(time.time() - t_infer, 3)  # wall-clock time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())