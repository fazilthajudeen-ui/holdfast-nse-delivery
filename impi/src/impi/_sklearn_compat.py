"""Keep scikit-learn importable when Windows Application Control blocks one of its compiled helpers.

On 11 Oct 2026 Smart App Control started blocking
`sklearn/metrics/_pairwise_distances_reduction/_datasets_pair.cp312-win_amd64.pyd` (unchanged since install on
10 Oct). scikit-learn imports that package eagerly from `sklearn.metrics` (and so from `linear_model`, `impute`,
...), although IMPI never uses it: it only accelerates nearest-neighbour / pairwise-distance computations.

We do NOT touch security settings and the blocked file is never loaded. If (and only if) importing the package
fails with an Application Control error, a placeholder module is registered under the same name. Its dispatchers
report `is_usable_for(...) == False`, which makes scikit-learn use its standard (non-accelerated) code paths;
anything that would still call them raises a clear error instead of silently computing something else.
"""
from __future__ import annotations

import sys
import types

NAME = "sklearn.metrics._pairwise_distances_reduction"
ACTIVE = False


class _Blocked:
    @classmethod
    def is_usable_for(cls, *args, **kwargs) -> bool:
        return False

    @classmethod
    def valid_metrics(cls) -> list:
        return []

    @classmethod
    def compute(cls, *args, **kwargs):
        raise RuntimeError(f"{NAME} is blocked by Windows Application Control on this machine")


def _sqeuclidean_row_norms(X, num_threads=None):
    import numpy as np
    X = np.asarray(X)
    return np.einsum("ij,ij->i", X, X)


def install() -> bool:
    global ACTIVE
    if NAME in sys.modules and not ACTIVE:
        return False
    try:
        __import__(NAME)
        return False
    except ImportError as e:
        if "Application Control" not in str(e):
            raise
    for k in [k for k in sys.modules if k == NAME or k.startswith(NAME + ".") or k.startswith("sklearn.metrics")]:
        del sys.modules[k]
    mod = types.ModuleType(NAME)
    mod.__doc__ = "placeholder: blocked by Windows Application Control (see impi._sklearn_compat)"
    mod.__path__ = []
    for n in ["ArgKmin", "ArgKminClassMode", "BaseDistancesReductionDispatcher", "RadiusNeighbors", "RadiusNeighborsClassMode"]:
        setattr(mod, n, type(n, (_Blocked,), {}))
    mod.sqeuclidean_row_norms = _sqeuclidean_row_norms
    mod.__all__ = ["ArgKmin", "ArgKminClassMode", "BaseDistancesReductionDispatcher", "RadiusNeighbors",
                   "RadiusNeighborsClassMode", "sqeuclidean_row_norms"]
    sys.modules[NAME] = mod
    ACTIVE = True
    return True
