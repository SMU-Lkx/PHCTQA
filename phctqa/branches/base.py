from pathlib import Path
from typing import Any, Dict, Optional
import torch

from phctqa.util.infer_util import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

def resolve_repo_path(p):
    if p is None:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p

def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "npu":
        import torch_npu  # direct import; missing torch_npu should fail loudly
        return torch.device("npu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("PHCTQA supports only device: cpu or cuda or npu")

class BaseBranchAdapter:
    def __init__(self, name: str, branch_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.name = name
        self.branch_cfg = branch_cfg
        self.global_cfg = global_cfg
        self.device = select_device(global_cfg["device"])
        self.threshold = branch_cfg.get("threshold", 0.5)
        train_config = branch_cfg.get("train_config")
        self.train_cfg = load_yaml(resolve_repo_path(train_config)) if train_config else {}
        self.weights_path = resolve_repo_path(branch_cfg.get("weights_path"))
        self.model = None

    def require_weights(self) -> Path:
        if self.weights_path is None:
            raise ValueError(f"{self.name}: weights is required for deep branch")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"{self.name}: weight file not found: {self.weights_path}")
        return self.weights_path

    def load(self) -> None:
        raise NotImplementedError(f"{self.name}: load() not implemented")

    def predict(self, input_path: str) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name}: predict() not implemented")

    @staticmethod
    def result(probability: float, threshold: Optional[float]):
        if threshold is None:
            return {"probability": float(probability), "pred": None, "threshold": None, "status": "ok"}
        return {
            "probability": float(probability),
            "pred": int(probability >= threshold),
            "threshold": float(threshold),
            "status": "ok",
        }

