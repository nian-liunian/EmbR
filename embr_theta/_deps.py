"""Check that EmbR pipeline dependency modules are importable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from embr_theta._bootstrap import project_root, setup_paths

_THETA = Path(__file__).resolve().parent
_ROOT = project_root()

PRECOMPUTE_MODULES = (
    "embr_io",
    "embr_features",
    "embr_dataset",
)

SCF_MODULES = (
    "scf_embed_io",
    "scf_embed_pyscf",
)

TRAIN_MODULES = (
    "soap_mm_util",
)


def _find_file(name: str) -> Path | None:
    for base in (_THETA, _ROOT):
        p = base / f"{name}.py"
        if p.is_file():
            return p
    return None


def require_modules(names: tuple[str, ...], *, stage: str) -> None:
    setup_paths()
    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return

    lines = [
        f"[embr_theta] missing Python modules for stage={stage!r}:",
        *[f"  - {n}.py" for n in missing],
        "",
        f"Expected under {_ROOT}/ or {_THETA}/",
    ]
    for n in missing:
        found = _find_file(n)
        lines.append(f"  {n}.py -> {found if found else 'NOT FOUND'}")
    raise SystemExit("\n".join(lines))


def require_precompute() -> None:
    require_modules(PRECOMPUTE_MODULES, stage="precompute")


def require_scf() -> None:
    require_modules(PRECOMPUTE_MODULES + SCF_MODULES, stage="line_rho")


def require_train() -> None:
    require_modules(PRECOMPUTE_MODULES + TRAIN_MODULES, stage="train")
