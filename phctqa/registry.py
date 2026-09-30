"""Branch registry for PHCTQA.

Each adapter is lazily imported so reviewers can run smoke tests even before all
project-local training modules are vendored into the repository.
"""
from dataclasses import dataclass
from importlib import import_module
from typing import Optional


@dataclass(frozen=True)
class BranchSpec:
    name: str
    type: str
    # config: Optional[str]
    weights: Optional[str]
    enabled: bool = True


BRANCH_TYPES = {
    "tool": "phctqa.branches.tool:ToolAdapter",
    "cnn":  "phctqa.branches.cnn:CNNAdapter",
    "mil":  "phctqa.branches.mil:MILAdapter",
}

def import_adapter(dotted: str):
    module_name, class_name = dotted.split(":")
    module = import_module(module_name)
    return getattr(module, class_name)
