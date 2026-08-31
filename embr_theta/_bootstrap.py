"""
Make ``embr_theta/`` runnable both ways:

  cd <project_root>     # parent of embr_theta/ ; must contain soap_e0_*.py, scf_embed_*.py, ...
  python -m embr_theta.precompute_soap --manifest embr_theta/manifest.json --out embr_theta/soap.npz

  cd <project_root>/theta
  python precompute_soap.py --manifest manifest.json --out soap.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

_THETA_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THETA_DIR.parent


def setup_paths() -> Path:
    """Insert embr_theta/ first, then project root, onto sys.path."""
    for p in (_THETA_DIR, _PROJECT_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return _PROJECT_ROOT


def project_root() -> Path:
    return _PROJECT_ROOT
