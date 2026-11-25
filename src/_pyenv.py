"""Use the IMPI venv's scientific stack (numpy / pandas / pyarrow) from the main Python install.

On 12 Oct 2026 Windows Smart App Control started blocking pyarrow's Parquet DLL in the main Python install
(pyarrow 25.0.1), so Delivery Lens could no longer read or write its .parquet files. The IMPI venv's
pyarrow 26 is not blocked. Security settings are not touched: we only put the venv's site-packages first on
sys.path, so numpy, pandas and pyarrow come from the venv as one consistent set, while packages the venv
does not have (fastapi, uvicorn, plotly, streamlit) still come from the main install.

Imported by src/__init__.py and app/__init__.py, i.e. before any module imports numpy or pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENV_SITE = Path(__file__).resolve().parents[1] / "impi" / ".venv" / "Lib" / "site-packages"


def use_venv_stack() -> bool:
    if not VENV_SITE.exists() or str(VENV_SITE) in sys.path:
        return False
    if any(m in sys.modules for m in ("numpy", "pandas", "pyarrow")):
        return False                       # too late to switch safely; leave the interpreter as it is
    sys.path.insert(0, str(VENV_SITE))
    return True
