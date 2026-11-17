"""Registry of the shipped directed-connectivity estimators, keyed by method name."""

from __future__ import annotations

from dirconn.methods.base import Estimator, MethodInfo
from dirconn.methods.coherency import PSI, ImCoh
from dirconn.methods.granger import GC
from dirconn.methods.multitaper_granger import MTGC
from dirconn.methods.pcmci import PCMCI
from dirconn.methods.spectral_estimators import DTF, GPDC, PDC, RPDC, SSGC
from dirconn.methods.time_reversal import TRGC
from dirconn.methods.timevarying.timevarying import ADTF, TPDC
from dirconn.methods.timevarying.windowed import WDTF
from dirconn.methods.transfer_entropy import TE

__all__ = ["REGISTRY", "get_estimator", "list_methods"]

REGISTRY: dict[str, type[Estimator]] = {
    "gc": GC,
    "dtf": DTF,
    "pdc": PDC,
    "gpdc": GPDC,
    "rpdc": RPDC,
    "ssgc": SSGC,
    "mtgc": MTGC,
    "imcoh": ImCoh,
    "psi": PSI,
    "trgc": TRGC,
    "te": TE,
    "pcmci": PCMCI,
    "adtf": ADTF,
    "tpdc": TPDC,
    "wdtf": WDTF,
}


def get_estimator(name: str) -> type[Estimator]:
    """Return the estimator class registered under ``name``.

    Parameters
    ----------
    name : str
        Method name, equal to the ``info.name`` of the estimator (for example ``"gc"``).

    Returns
    -------
    type of Estimator
        The estimator class.

    Raises
    ------
    KeyError
        If ``name`` is not registered; the message lists the known names.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown method {name!r}; known methods: {sorted(REGISTRY)}.") from None


def list_methods() -> list[MethodInfo]:
    """Return the :class:`~dirconn.methods.base.MethodInfo` of every method, sorted by name.

    Returns
    -------
    list of MethodInfo
        One entry per registered method.
    """
    return [REGISTRY[name].info for name in sorted(REGISTRY)]
