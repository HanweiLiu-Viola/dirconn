"""Helpers shared by the resampling statistics (surrogate test, bootstrap, permutation)."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from dirconn.exceptions import DirconnWarning
from dirconn.provenance import AssumptionCheck
from dirconn.result import Connectivity

Reduction = Callable[[Connectivity], Connectivity] | str | Sequence[float] | None


def make_reducer(reduce: Reduction) -> Callable[[Connectivity], Connectivity]:
    """Turn the ``reduce=`` argument into a function applied to every fit."""
    if reduce is None:
        return lambda result: result
    if callable(reduce):
        return reduce
    band: str | Sequence[float] = reduce
    return lambda result: result.reduce_freq(band)


def without_data_checks(estimator: Any) -> Any:
    """Return ``estimator`` with the order-stability and whiteness checks off, for resampled fits.

    Both judge the model fitted to the data: the order-stability check refits it at several longer
    orders, and the whiteness check asks whether its order leaves autocorrelated residuals. They
    belong to the fit on the data, not to every surrogate, resample or relabelling (the residuals
    of a fit on phase-randomised surrogates, for example, are less white than those on the data).
    Parameters an estimator does not have are left alone.
    """
    names = (
        {f.name for f in dataclasses.fields(estimator)}
        if dataclasses.is_dataclass(estimator)
        else set()
    )
    switches = {
        name: False
        for name in ("check_order_stability", "check_whiteness")
        if name in names and getattr(estimator, name)
    }
    return dataclasses.replace(estimator, **switches) if switches else estimator


def sort_warnings(
    caught: list[warnings.WarningMessage],
    other: dict[tuple[type[Warning], str], warnings.WarningMessage],
) -> list[str]:
    """Return the dirconn diagnostics as text; keep other warnings once so they can be re-issued."""
    notes: list[str] = []
    for item in caught:
        if issubclass(item.category, DirconnWarning):
            notes.append(str(item.message))
        else:
            other.setdefault((item.category, str(item.message)), item)
    return notes


def reissue(other: dict[tuple[type[Warning], str], warnings.WarningMessage]) -> None:
    """Emit each collected non-dirconn warning once."""
    for message in other.values():
        warnings.warn_explicit(message.message, message.category, message.filename, message.lineno)


def summary_check(name: str, notes: list[str], n_runs: int, what: str) -> AssumptionCheck:
    """Summarise per-run diagnostics in one check and one warning.

    ``notes`` holds the first message of every run that raised a diagnostic.
    """
    if not notes:
        return AssumptionCheck(name, "pass", detail=f"{what} raised no diagnostics.")
    flagged = len(notes)
    detail = (
        f"{flagged} of {n_runs} runs of {what} raised a diagnostic (first: {notes[0]}). "
        "The diagnostics of the fit on the data itself are reported separately."
    )
    warnings.warn(detail, DirconnWarning, stacklevel=3)
    return AssumptionCheck(name, "warn", detail=detail, value=float(flagged))


def power_check(
    n_defined: int, n_resamples: int, alpha: float, *, check: str, argument: str
) -> AssumptionCheck:
    """Check that the rank p-value floor ``1 / (n + 1)`` can reach ``alpha / n_defined``.

    ``check`` names the entry in the checklist and ``argument`` the user-facing parameter
    (``"n_surrogates"``, ``"n_permutations"``) used in the message.
    """
    required = math.ceil(n_defined / alpha - 1)
    if n_resamples < required:
        detail = (
            f"{argument}={n_resamples} is below {required}: with {n_defined} defined cells "
            f"and alpha={alpha:g} the smallest p-value 1/({argument} + 1) cannot reach "
            f"alpha/{n_defined}, so a Bonferroni-type correction can never reject. Increase "
            f"{argument} or reduce the number of tested cells with reduce=."
        )
        warnings.warn(detail, DirconnWarning, stacklevel=3)
        return AssumptionCheck(check, "warn", detail=detail, value=float(required))
    return AssumptionCheck(
        check,
        "pass",
        detail=f"{argument}={n_resamples} covers {n_defined} defined cells at alpha={alpha:g}.",
        value=float(required),
    )
