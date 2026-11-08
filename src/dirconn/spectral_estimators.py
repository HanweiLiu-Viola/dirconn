"""DTF, PDC, gPDC, rPDC and SSGC: spectral directed-influence measures from a pooled VAR model."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np

from dirconn.estimator import (
    Estimator,
    MethodInfo,
    check_switches,
    order_boundary_check,
    residual_whiteness_check,
    sampling_check,
    stability_check,
)
from dirconn.mvar import VarFit, check_order_spec, choose_order, fit_var, row_mask
from dirconn.provenance import STANDARD_ASSUMPTIONS
from dirconn.result import Connectivity
from dirconn.spectral import (
    dtf_from_coefs,
    frequency_grid,
    pdc_from_coefs,
    rank_one_freqs,
    rpdc_from_coefs,
)
from dirconn.statespace import spectral_state_space_gc
from dirconn.timeseries import TimeSeries


def _standardise(data: TimeSeries) -> tuple[TimeSeries, dict[str, Any]]:
    """Divide every channel by its standard deviation over all trials and samples.

    The mean is not subtracted here: centring is the job of ``demean`` in the VAR fit, so with the
    default settings the result is the z-score. With a ``valid`` mask the standard deviation is
    that of the valid samples; the masks are kept.
    """
    x = np.asarray(data.data)
    if data.valid is None:
        scale = (x - x.mean(axis=(0, 2), keepdims=True)).std(axis=(0, 2))
    else:
        scale = np.moveaxis(x, 1, 0)[:, data.valid].std(axis=1)
    flat = np.flatnonzero(scale == 0)
    if flat.size:
        raise ValueError(
            f"Channels {[data.names[i] for i in flat]} have zero variance and cannot be "
            "standardised."
        )
    series = replace(data, data=x / scale[None, :, None])
    return series, {"standardized": True, "scale": scale.tolist()}


@dataclass(frozen=True)
class _SpectralVAR(Estimator):
    """Shared machinery: fit one pooled VAR model, then evaluate a spectral measure on it.

    The order is chosen once, on the model of all channels. The frequency axis runs from 0 to the
    Nyquist frequency by default; see :func:`dirconn.spectral.frequency_grid`. Data with ``valid``
    or ``predict`` masks use only the rows those select (see :func:`dirconn.mvar.row_mask`).
    """

    _needs_stable_model: ClassVar[bool] = False
    """Raise instead of only warning when the stationarity check fails."""

    order: int | str = "bic"
    max_order: int = 10
    n_freqs: int = 128
    fmax: float | None = None
    freqs: Any = None
    ridge: float = 0.0
    demean: bool = True
    n_folds: int = 5
    check_order_stability: bool = True
    check_whiteness: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        check_order_spec(self.order, self.max_order, self.ridge, self.n_folds)
        check_switches(self, "check_order_stability", "check_whiteness")
        if self.freqs is not None:
            if self.fmax is not None:
                raise ValueError("Pass either freqs or fmax, not both.")
            return
        if (
            isinstance(self.n_freqs, bool)
            or not isinstance(self.n_freqs, (int, np.integer))
            or self.n_freqs < 2
        ):
            raise ValueError(f"n_freqs must be an integer of at least 2, got {self.n_freqs!r}.")
        if self.fmax is not None and (not math.isfinite(float(self.fmax)) or self.fmax <= 0):
            raise ValueError(f"fmax must be a positive finite number, got {self.fmax!r}.")

    def _preprocess(self, data: TimeSeries) -> tuple[TimeSeries, dict[str, Any]]:
        """Return the series that is fitted and the preprocessing to record; default: none."""
        return data, {}

    @abstractmethod
    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        """Return the measure with layout ``[source, target, freq]``."""

    def _measure_details(self, fit: VarFit, freqs: np.ndarray) -> dict[str, Any]:
        """Return method-specific entries for the provenance details; default: none."""
        return {}

    def _conditional_order_stability(self) -> bool:
        """Whether the order-stability check compares with conditional GC; default: yes."""
        return True

    def _fit(self, data: TimeSeries) -> Connectivity:
        freqs = frequency_grid(data.sfreq, n_freqs=self.n_freqs, fmax=self.fmax, freqs=self.freqs)
        series, preprocessing = self._preprocess(data)
        order, selection = choose_order(
            series,
            self.order,
            self.max_order,
            ridge=self.ridge,
            demean=self.demean,
            n_folds=self.n_folds,
        )
        fit = fit_var(series, order, ridge=self.ridge, demean=self.demean)
        sampling = sampling_check(fit.n_channels * order, fit.n_obs, stacklevel=4)
        stability = stability_check(fit.coefs, stacklevel=4)
        if self._needs_stable_model and stability.status == "fail":
            raise ValueError(
                f"{self.info.name} needs a stable VAR model, but {stability.detail} Check the "
                "data for trends or non-stationarity, or lower the order."
            )
        values = self._measure(fit, freqs)

        checks = [sampling, stability]
        if self.check_whiteness:
            checks.append(
                residual_whiteness_check(
                    fit.residuals,
                    series.n_trials,
                    order,
                    rows=row_mask(series.valid, series.predict, order, series.n_times),
                    stacklevel=4,
                )
            )
        if selection is not None:
            checks.append(order_boundary_check(selection, stacklevel=4))
        if self.check_order_stability:
            from dirconn.order_stability import order_stability_check

            checks.append(
                order_stability_check(
                    series,
                    order,
                    demean=self.demean,
                    conditional=self._conditional_order_stability(),
                    stacklevel=4,
                )
            )
        provenance = self._provenance(
            data,
            model_order=order,
            preprocessing=preprocessing,
            details={
                "n_obs": fit.n_obs,
                "spectral_radius": stability.value,
                "order_selection": None if selection is None else selection.to_dict(),
                **self._measure_details(fit, freqs),
            },
            checks=checks,
        )
        return Connectivity.from_array(
            values, source=list(data.names), freqs=freqs, provenance=provenance
        )


_BACKEND = "own numpy pooled OLS VAR, vectorised frequency response"


@dataclass(frozen=True)
class DTF(_SpectralVAR):
    """Directed transfer function (Kaminski and Blinowska), a predictive spectral measure.

    ``values[i, j, f]`` is the share of the signal reaching channel ``j`` at frequency ``f`` that
    comes from channel ``i``: the squared transfer function normalised over the sources of each
    target, so the values over all sources sum to 1. It mixes direct and indirect paths. The
    diagonal (a channel's share of itself) is defined, unlike in :class:`GC`.

    Parameters
    ----------
    order : int or {"aic", "bic", "hqc", "cv"}
        Fixed VAR order, or the criterion that selects it once on the model of all channels.
    max_order : int
        Largest candidate order when ``order`` is a criterion.
    n_freqs, fmax, freqs
        Frequency axis: ``n_freqs`` points from 0 to ``fmax`` (default the Nyquist frequency), or
        an explicit ``freqs`` in Hz.
    ridge, demean, n_folds
        As for :func:`dirconn.fit_var` and :func:`dirconn.select_order`.
    standardize : bool
        Divide every channel by its standard deviation before fitting, because the result depends
        on the relative scale of the channels. The factors are recorded in the provenance.
    check_order_stability, check_whiteness : bool
        Run the order-stability check and the residual whiteness check, as for :class:`GC`. The
        resampling tests switch both off for the resampled fits.
    """

    info = MethodInfo(
        name="dtf",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=True,
        frequency_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    standardize: bool = True

    def _depends_on_scale(self) -> bool:
        return not self.standardize

    def _preprocess(self, data: TimeSeries) -> tuple[TimeSeries, dict[str, Any]]:
        if self.standardize:
            return _standardise(data)
        return data, {"standardized": False}

    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        return dtf_from_coefs(fit.coefs, freqs, fit.sfreq)


@dataclass(frozen=True)
class _PartialDirectedCoherence(_SpectralVAR):
    squared: bool = True


@dataclass(frozen=True)
class PDC(_PartialDirectedCoherence):
    """Partial directed coherence (Baccala and Sameshima), a predictive spectral measure.

    ``values[i, j, f]`` is the share of the outflow of channel ``i`` at frequency ``f`` that goes
    to channel ``j``, normalised over the targets of each source. Because it is built from the
    model coefficients rather than their inverse, it reflects direct influences only. The values
    depend on the relative scale of the channels; see :class:`GPDC` for the version that does not.
    The parameters not listed below are those of :class:`DTF`.

    Parameters
    ----------
    squared : bool
        ``True`` (default): the squared form, whose values sum to 1 over the targets. ``False``:
        the amplitude form, its square root.
    standardize : bool
        Divide every channel by its standard deviation before fitting. The factors are recorded
        in the provenance.
    """

    info = MethodInfo(
        name="pdc",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=True,
        frequency_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    standardize: bool = True

    def _depends_on_scale(self) -> bool:
        return not self.standardize

    def _preprocess(self, data: TimeSeries) -> tuple[TimeSeries, dict[str, Any]]:
        if self.standardize:
            return _standardise(data)
        return data, {"standardized": False}

    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        return pdc_from_coefs(fit.coefs, freqs, fit.sfreq, squared=self.squared)


@dataclass(frozen=True)
class GPDC(_PartialDirectedCoherence):
    """Generalised partial directed coherence (Baccala et al. 2007).

    Like :class:`PDC`, but every term is weighted by the inverse innovation variance of its target,
    so the result does not depend on the scale of the channels and needs no standardisation.

    The parameters are those of :class:`PDC` without ``standardize``.
    """

    info = MethodInfo(
        name="gpdc",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, 1.0),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        return pdc_from_coefs(
            fit.coefs, freqs, fit.sfreq, squared=self.squared, noise_var=np.diag(fit.noise_cov)
        )


@dataclass(frozen=True)
class RPDC(_SpectralVAR):
    """Renormalised partial directed coherence (Schelter et al. 2009), a predictive measure.

    ``values[source, target, f]`` is ``lambda`` of the influence of ``source`` on ``target``: the
    squared real and imaginary part of the AR polynomial entry ``Abar[target, source](f)``,
    weighted by the inverse of their asymptotic covariance (see
    :func:`dirconn.spectral.rpdc_from_coefs`). In the paper's indices (``i`` target, ``j``
    source) it is ``lambda_ij(f)``, stored at ``values[j, i, f]``. Unlike
    :class:`PDC` it is not normalised over the targets; it measures how far the direct influence
    at ``f`` is from zero relative to its estimation error, so it does not depend on the scale of
    the channels. With a real connection ``lambda`` tends to a value that does not depend on the
    recording length; ``N lambda`` is the Wald statistic that
    :func:`dirconn.analytic_rpdc_test` compares with the chi-square distribution.

    ``lambda`` resolves frequency only from order 3 on. The real and imaginary part are two
    linear combinations of the ``order`` coefficients of the pair, and a Wald statistic does not
    change under an invertible linear map: at order 1 ``lambda`` is the same at every frequency,
    and at order 2 it is the same at every frequency other than 0 and the Nyquist frequency, where
    it equals the joint test of both lags in the time domain.

    The model is fitted by ordinary least squares only: the asymptotic covariance holds for that
    fit, so there is no ``ridge`` and rPDC suits a few channels (regions or selected sensors), not
    a high-dimensional sensor-level fit. Like every MVAR measure it is affected by volume
    conduction and leakage: an instantaneous mixture of sources changes the coefficients.

    Parameters
    ----------
    order, max_order, n_freqs, fmax, freqs, demean, n_folds
        As for :class:`DTF`.
    check_order_stability, check_whiteness
        As for :class:`DTF`.
    """

    info = MethodInfo(
        name="rpdc",
        backend=_BACKEND,
        directed=True,
        min_nodes=2,
        value_range=(0.0, math.inf),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    ridge: float = field(default=0.0, init=False, repr=False)

    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        values, _ = rpdc_from_coefs(fit.coefs, fit.noise_cov, fit.regressor_cov, freqs, fit.sfreq)
        return values

    def _measure_details(self, fit: VarFit, freqs: np.ndarray) -> dict[str, Any]:
        single = rank_one_freqs(fit.order, freqs, fit.sfreq)
        return {"rank_one_freqs": [float(f) for f in np.asarray(freqs)[single]]}


@dataclass(frozen=True)
class SSGC(_SpectralVAR):
    """Spectral Granger causality from the state-space form of the VAR (Barnett and Seth 2015).

    ``values[i, j, f]`` is the Geweke spectral Granger causality from channel ``i`` to channel
    ``j`` at frequency ``f``, in nats. One VAR model of all channels is fitted; written in
    innovations form, it gives every reduced model exactly by one discrete algebraic Riccati
    equation per conditioning set (see :mod:`dirconn.statespace`), so no reduced model is refitted
    and no non-parametric spectral factorisation is needed. With ``conditional=True`` (default)
    the influence of ``i`` on ``j`` is conditioned on all other channels,
    ``f_{i->j|rest}(f)``; with ``conditional=False`` each pair is judged from its bivariate
    subprocess of the same fit, which for two channels is Geweke's (1982) formula.

    The mean over the whole unit circle (the trapezoid mean over the default grid from 0 to the
    Nyquist frequency) is the time-domain value of ``GC(method="state_space")``. The values do not
    depend on the scale of the channels. Unlike :class:`MTGC` the measure is parametric and
    conditional, so it suits short segments and more than two channels; like every MVAR measure
    it needs a stationary, well-specified model, and an unstable fit raises ``ValueError``.

    Parameters
    ----------
    order, max_order, n_freqs, fmax, freqs, ridge, demean, n_folds
        As for :class:`DTF`. With ``ridge > 0`` the penalised fit and its residual covariance
        define the model.
    conditional : bool
        Condition on all other channels (default) or judge channel pairs separately. Costs one
        Riccati equation per channel, or one per pair and per channel.
    check_order_stability, check_whiteness : bool
        As for :class:`DTF`; the order-stability check compares with least-squares GC refits in
        the same mode (conditional or pairwise).
    """

    info = MethodInfo(
        name="ssgc",
        backend="own numpy pooled OLS VAR, innovations-form state space, scipy DARE",
        directed=True,
        min_nodes=2,
        value_range=(0.0, math.inf),
        scale_dependent=False,
        frequency_resolved=True,
        assumptions=STANDARD_ASSUMPTIONS,
        supports_mask=True,
    )

    _needs_stable_model: ClassVar[bool] = True

    conditional: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        super().__post_init__()
        check_switches(self, "conditional")

    def _conditional_order_stability(self) -> bool:
        return self.conditional

    def _measure(self, fit: VarFit, freqs: np.ndarray) -> np.ndarray:
        return spectral_state_space_gc(
            fit.coefs, fit.noise_cov, freqs, fit.sfreq, conditional=self.conditional
        )

    def _measure_details(self, fit: VarFit, freqs: np.ndarray) -> dict[str, Any]:
        return {"conditional": self.conditional}


def dtf(data: TimeSeries, **params: Any) -> Connectivity:
    """Directed transfer function of ``data``; shorthand for ``DTF(**params).fit(data)``."""
    return DTF(**params).fit(data)


def pdc(data: TimeSeries, **params: Any) -> Connectivity:
    """Partial directed coherence of ``data``; shorthand for ``PDC(**params).fit(data)``."""
    return PDC(**params).fit(data)


def gpdc(data: TimeSeries, **params: Any) -> Connectivity:
    """Generalised PDC of ``data``; shorthand for ``GPDC(**params).fit(data)``."""
    return GPDC(**params).fit(data)


def rpdc(data: TimeSeries, **params: Any) -> Connectivity:
    """Renormalised PDC of ``data``; shorthand for ``RPDC(**params).fit(data)``."""
    return RPDC(**params).fit(data)


def ssgc(data: TimeSeries, **params: Any) -> Connectivity:
    """State-space spectral GC of ``data``; shorthand for ``SSGC(**params).fit(data)``."""
    return SSGC(**params).fit(data)
