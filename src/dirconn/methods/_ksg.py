"""KSG estimator of conditional mutual information, on the CPU (scipy) or with PyTorch.

The CPU path is the reference: scipy's ``cKDTree`` with the maximum norm. The torch path computes
the same neighbour distances and counts by brute force in blocks of rows, on any torch device
(``"cpu"``, ``"cuda"``, ``"xpu"``, ``"mps"``). In float64 both give the same counts, so the
estimates agree to rounding of the digamma sums; float32 is faster on most GPUs but changes
neighbours whose distances tie at float32 precision, so it only agrees statistically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_STRICT = 1.0 - 1e-9
"""Neighbours are counted strictly inside the k-th neighbour distance, as in tigramite."""

DTYPES: tuple[str, ...] = ("float64", "float32")

_BLOCK_BYTES = 256 * 2**20
"""Memory for one block of the brute-force distance matrices on the torch device."""


def _columns(values: Any, name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or (n is not None and arr.shape[0] != n):
        raise ValueError(f"{name} must have shape (n_samples, n_dims) with {n} samples.")
    return arr


def _standardise(arr: np.ndarray) -> np.ndarray:
    std = arr.std(axis=0)
    if np.any(std == 0):
        raise ValueError("A variable of the mutual information is constant.")
    return (arr - arr.mean(axis=0)) / std


def _workers(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or (value < 1 and value != -1)
    ):
        raise ValueError(f"workers must be a positive integer or -1, got {value!r}.")
    return int(value)


def check_device(device: Any, dtype: Any) -> None:
    """Validate a ``device`` and ``dtype`` pair; ``device=None`` is the scipy path (float64)."""
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError(f"device must be None or a torch device name, got {device!r}.")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}.")
    if device is None and dtype != "float64":
        raise ValueError("dtype='float32' needs a torch device; the scipy path is float64.")


def require_torch() -> Any:
    """Import torch, or raise an ``ImportError`` that names the extra to install."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "The KSG estimator on a torch device needs PyTorch: install it with "
            "`pip install dirconn[gpu]` (see pytorch.org for a CUDA or XPU build)."
        ) from None
    except OSError as error:  # a DLL of torch that does not load, seen on Windows
        raise ImportError(
            f"PyTorch is installed but could not be loaded ({error}). On Windows this happens "
            "when numpy from conda (MKL) has already run linear algebra in the process and torch "
            "comes from pip: import torch first, or install both from the same source."
        ) from error
    return torch


def ksg_cmi(
    x: Any,
    y: Any,
    z: Any = None,
    *,
    k: int = 4,
    workers: int = 1,
    device: str | None = None,
    dtype: str = "float64",
) -> float:
    """Conditional mutual information ``I(X; Y | Z)`` in nats, KSG estimator (Frenzel-Pompe).

    Every variable is standardised; distances use the maximum norm. For every sample the distance
    to its ``k``-th neighbour in the joint space sets the radius within which neighbours are
    counted in the subspaces ``(X, Z)``, ``(Y, Z)`` and ``Z``:
    ``psi(k) - mean(psi(n_xz) + psi(n_yz) - psi(n_z))``, the counts including the sample itself.
    Without ``Z`` this is Kraskov's first mutual information estimator.

    Parameters
    ----------
    x, y : array_like, shape (n_samples,) or (n_samples, n_dims)
    z : array_like, optional
        The conditioning variables, same number of samples.
    k : int
        Number of neighbours in the joint space.
    workers : int
        Processes for the neighbour searches (``-1``: all cores), passed to scipy's ``cKDTree``.
        The estimate does not depend on it; ignored on a torch device.
    device : str, optional
        ``None`` (default) uses scipy's ``cKDTree`` on the CPU. A torch device name (``"cpu"``,
        ``"cuda"``, ``"cuda:1"``, ``"xpu"``, ``"mps"``) computes the neighbours by brute force
        with PyTorch (extra ``gpu``).
    dtype : {"float64", "float32"}
        Precision of the distances on a torch device. float64 gives the counts of the scipy path;
        float32 is faster on most GPUs but only agrees statistically.

    Returns
    -------
    float
        The estimate; it can be slightly negative when the true value is near zero.
    """
    return float(ksg_cmi_many([x], y, z, k=k, workers=workers, device=device, dtype=dtype)[0])


def ksg_cmi_many(
    xs: Sequence[Any],
    y: Any,
    z: Any = None,
    *,
    k: int = 4,
    workers: int = 1,
    device: str | None = None,
    dtype: str = "float64",
) -> np.ndarray:
    """``I(X; Y | Z)`` for every ``X`` in ``xs`` with the same ``Y`` and ``Z``.

    The greedy selection of multivariate transfer entropy evaluates many candidates (and their
    surrogates) against one target and one condition; this shares the work that depends only on
    ``Y`` and ``Z``. Each value equals :func:`ksg_cmi` of that ``X``.

    Parameters
    ----------
    xs : sequence of array_like
        The variables ``X``, each of shape (n_samples,) or (n_samples, n_dims).
    y, z, k, workers, device, dtype
        As in :func:`ksg_cmi`.

    Returns
    -------
    ndarray, shape (len(xs),)
    """
    from scipy.special import digamma

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    workers = _workers(workers)
    check_device(device, dtype)
    if len(xs) == 0:
        return np.empty(0)
    first = _columns(xs[0], "x")
    n = first.shape[0]
    if n <= k + 1:
        raise ValueError(f"Need more than k + 1 = {k + 1} samples, got {n}.")
    ys = _standardise(_columns(y, "y", n))
    zs = None if z is None else _standardise(_columns(z, "z", n))
    if zs is not None and zs.shape[1] == 0:
        zs = None
    standardised = [_standardise(_columns(x, "x", n)) for x in xs]
    if device is None:
        counts = _counts_scipy(standardised, ys, zs, k, workers)
    else:
        counts = _counts_torch(standardised, ys, zs, k, device, dtype)
    out = np.empty(len(xs))
    for i, (n_xz, n_yz, n_z) in enumerate(counts):
        out[i] = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(n_z))
    return out


def _counts_scipy(
    xs: Sequence[np.ndarray], y: np.ndarray, z: np.ndarray | None, k: int, workers: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from scipy.spatial import cKDTree

    n = y.shape[0]
    yz = y if z is None else np.hstack([y, z])
    yz_tree = cKDTree(yz)
    z_tree = None if z is None else cKDTree(z)

    def count(tree: Any, space: np.ndarray, radius: np.ndarray) -> np.ndarray:
        return np.asarray(
            tree.query_ball_point(space, r=radius, p=np.inf, return_length=True, workers=workers)
        )

    out = []
    for x in xs:
        joint = np.hstack([x, yz])
        distance = cKDTree(joint).query(joint, k=[k + 1], p=np.inf, workers=workers)[0][:, 0]
        distance = distance * _STRICT
        xz = x if z is None else np.hstack([x, z])
        n_xz = count(cKDTree(xz), xz, distance)
        n_yz = count(yz_tree, yz, distance)
        n_z = np.full(n, n) if z is None else count(z_tree, z, distance)
        out.append((n_xz, n_yz, n_z))
    return out


def _counts_torch(
    xs: Sequence[np.ndarray],
    y: np.ndarray,
    z: np.ndarray | None,
    k: int,
    device: str,
    dtype: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    torch = require_torch()
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    dev = torch.device(device)
    n = y.shape[0]
    itemsize = 8 if dtype == "float64" else 4
    # about six block x n matrices live at once (three distances, their maxima, a mask)
    block = max(1, min(n, _BLOCK_BYTES // (6 * n * itemsize)))
    inf = float("inf")
    ty = torch.as_tensor(y, dtype=tdtype, device=dev)
    tz = None if z is None else torch.as_tensor(z, dtype=tdtype, device=dev)
    out = []
    with torch.no_grad():
        for x in xs:
            tx = torch.as_tensor(x, dtype=tdtype, device=dev)
            n_xz = torch.empty(n, dtype=torch.int64, device=dev)
            n_yz = torch.empty(n, dtype=torch.int64, device=dev)
            n_z = torch.full((n,), n, dtype=torch.int64, device=dev)
            for start in range(0, n, block):
                rows = slice(start, min(n, start + block))
                dx = torch.cdist(tx[rows], tx, p=inf)
                dy = torch.cdist(ty[rows], ty, p=inf)
                dz = None if tz is None else torch.cdist(tz[rows], tz, p=inf)
                dyz = dy if dz is None else torch.maximum(dy, dz)
                dxz = dx if dz is None else torch.maximum(dx, dz)
                joint = torch.maximum(dx, dyz)
                # the (k + 1)-th smallest distance: the sample itself is at distance 0
                kth = torch.kthvalue(joint, k + 1, dim=1).values[:, None]
                if dtype == "float64":
                    radius = kth * _STRICT
                    n_xz[rows] = (dxz <= radius).sum(dim=1)
                    n_yz[rows] = (dyz <= radius).sum(dim=1)
                    if dz is not None:
                        n_z[rows] = (dz <= radius).sum(dim=1)
                else:  # 1 - 1e-9 rounds to 1 in float32: count strictly inside instead
                    n_xz[rows] = (dxz < kth).sum(dim=1)
                    n_yz[rows] = (dyz < kth).sum(dim=1)
                    if dz is not None:
                        n_z[rows] = (dz < kth).sum(dim=1)
            out.append((n_xz.cpu().numpy(), n_yz.cpu().numpy(), n_z.cpu().numpy()))
    return out
