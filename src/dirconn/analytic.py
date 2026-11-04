"""Analytic significance: the F test of Granger causality and the asymptotic tests of PDC and rPDC.

The Granger value ``log(SSE_reduced / SSE_full)`` compares two nested least-squares models, so the
classical F test applies exactly:

    F = (exp(GC) - 1) * df2 / df1,    df1 = order,    df2 = n_obs - n_regressors - (1 if demeaned)

with ``n_regressors = n_channels * order`` for the conditional model and ``2 * order`` for the
pairwise one.

PDC and rPDC are tested per frequency with the large-sample distribution of the least-squares
coefficients (Schelter et al. 2006, 2009): :func:`analytic_pdc_test` and
:func:`analytic_rpdc_test`. DTF, gPDC and the other measures have no such test here; use
:func:`dirconn.surrogate_test` or :func:`dirconn.bootstrap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from dirconn.correction import Correctable, check_alpha
from dirconn.granger import GC
from dirconn.mvar import VarFit, fit_var
from dirconn.result import Connectivity
from dirconn.spectral import ar_polynomial, pdc_null_scale, rpdc_from_coefs
from dirconn.spectral_estimators import PDC, RPDC, _SpectralVAR
from dirconn.timeseries import TimeSeries

NULL = (
    "the past of the source adds no predictive information about the target beyond the past of "
    "the target (and, in conditional mode, of the other channels)"
)
LIMITATIONS: tuple[str, ...] = (
    "The F distribution is exact only for a stationary linear VAR with independent, "
    "homoscedastic Gaussian residuals; heavy tails, bursts or changing variance make the "
    "p-values wrong in either direction.",
    "The p-values are conditional on the chosen model order. When the order is selected from "
    "the data (a criterion) the selection uncertainty is ignored; a too-low order leaves "
    "autocorrelated residuals, which inflates significance.",
    "Trials are pooled as independent observations of one VAR model; trial-to-trial changes in "
    "the process or serial dependence between trials are not accounted for.",
    "Predictive influence is tested, not causation: common input, a hidden driver or volume "
    "conduction can produce a significant value.",
)


@dataclass(frozen=True, eq=False)
class AnalyticTestResult(Correctable):
    """F-test p-values of a Granger result, with the premise they rest on.

    Attributes
    ----------
    observed : Connectivity
        The Granger result that was tested, with its provenance.
    pvalues, statistic : ndarray
        Shape ``(source, target, 1, 1)``; the F statistic and its upper-tail p-value. Undefined
        cells (the diagonal) are NaN.
    df : tuple of int
        Numerator and denominator degrees of freedom.
    null : str
        The null hypothesis.
    limitations : tuple of str
        Ways the p-values can be wrong or misleading.
    alpha : float
        Default level for :meth:`significant`.
    """

    observed: Connectivity
    pvalues: np.ndarray
    statistic: np.ndarray
    df: tuple[int, int]
    null: str
    limitations: tuple[str, ...]
    alpha: float

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("pvalues", "statistic"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator."""
        return self.observed.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a p-value."""
        return ~np.isnan(self.statistic)

    def __repr__(self) -> str:
        """Return a summary that includes the null hypothesis."""
        shape = "x".join(str(n) for n in self.pvalues.shape)
        return (
            f"AnalyticTestResult(method={self.method!r}, shape={shape}, "
            f"df={self.df}, null={self.null!r})"
        )


def analytic_gc_test(estimator: GC, data: TimeSeries, *, alpha: float = 0.05) -> AnalyticTestResult:
    """Test Granger causality with the F distribution of nested least-squares models.

    Parameters
    ----------
    estimator : GC
        Must have ``ridge=0``: with a penalty the models are not ordinary least squares and the
        F distribution does not apply.
    data : TimeSeries
        Input data.
    alpha : float
        Default level for :meth:`AnalyticTestResult.significant`.

    Returns
    -------
    AnalyticTestResult

    Raises
    ------
    TypeError
        If ``estimator`` is not a :class:`~dirconn.GC`.
    ValueError
        With a ridge penalty, or when the data leave no residual degrees of freedom.
    """
    if not isinstance(estimator, GC):
        raise TypeError(
            "analytic_gc_test is the F test of GC (nested least-squares models); "
            f"got {type(estimator).__name__}. PDC and RPDC have analytic_pdc_test and "
            "analytic_rpdc_test; for DTF, gPDC and the other methods use surrogate_test or "
            "bootstrap."
        )
    if estimator.ridge != 0.0:
        raise ValueError("The F test needs ordinary least squares: use ridge=0.")
    alpha = check_alpha(alpha)

    observed = estimator.fit(data)
    details = observed.provenance.details
    order = int(observed.provenance.model_order or 0)
    n_obs = int(details["n_obs"])
    n_channels = len(observed.source)
    n_regressors = n_channels * order if details["conditional"] else 2 * order
    df1 = order
    df2 = n_obs - n_regressors - (1 if estimator.demean else 0)
    if df2 < 1:
        raise ValueError(
            f"{n_obs} observations leave {df2} residual degree(s) of freedom for a model with "
            f"{n_regressors} regressors per equation; the F test needs at least 1."
        )

    values = observed.values
    with np.errstate(invalid="ignore"):
        statistic = np.expm1(values) * df2 / df1
    pvalues = np.where(np.isnan(statistic), np.nan, stats.f.sf(statistic, df1, df2))
    return AnalyticTestResult(
        observed=observed,
        pvalues=pvalues,
        statistic=statistic,
        df=(df1, df2),
        null=NULL,
        limitations=LIMITATIONS,
        alpha=alpha,
    )


SPECTRAL_NULL = (
    "at this frequency the source has no direct influence on the target in the fitted VAR model: "
    "Abar[target, source](f) = 0"
)
SPECTRAL_LIMITATIONS: tuple[str, ...] = (
    "The distribution is asymptotic (large N) for a stationary linear VAR fitted by ordinary least "
    "squares; with short recordings, heavy tails or changing variance the level can be off in "
    "either direction.",
    "The p-values are conditional on the model order. The uncertainty of a selected order is "
    "ignored; a too-low order leaves correlated residuals, which makes the test too permissive.",
    "Trials are pooled as independent realisations of one VAR process.",
    "Each frequency is tested on its own: a pair without an influence crosses the level at some "
    "frequency far more often than alpha. Correct over the pairs and frequencies of the result "
    "(adjusted()); test a band by giving the estimator that band's freqs or fmax.",
    "Predictive influence is tested, not causation: common input, a hidden driver or volume "
    "conduction can produce a significant value.",
    "Observation noise on coupled signals gives their best VAR small real direct terms where the "
    "network has no direct edge, and long recordings find them: in dirconn's calibration (a chain "
    "at 10 dB, order 0.2 s) 4 to 21 % of the absent direct edges were significant at 0.05, more "
    "with longer recordings and lower orders, as with surrogate tests (docs/statistics.md).",
)
PDC_LIMITATIONS: tuple[str, ...] = SPECTRAL_LIMITATIONS + (
    "Schelter et al. (2006) approximate N |Abar_ij|^2 / C_ij by chi-square with one degree of "
    "freedom; it is a weighted sum of two such variables (the real and the imaginary part). The "
    "approximation is exact at 0 Hz and the Nyquist frequency and conservative elsewhere: at "
    "alpha = 0.05 the rate under the null lies between 0.021 (equal weights) and 0.05, at "
    "alpha = 0.01 between 0.0013 and 0.01. analytic_rpdc_test uses the exact two-degree form.",
)


@dataclass(frozen=True, eq=False)
class AsymptoticTestResult(Correctable):
    """Per-frequency p-values and thresholds of PDC or rPDC, with the premise they rest on.

    Attributes
    ----------
    observed : Connectivity
        The result that was tested, with its provenance.
    pvalues, statistic, threshold : ndarray
        Shape ``(source, target, freq, 1)``. The statistic is ``N |Abar_ij(f)|^2 / C_ij(f)``
        (PDC) or ``N lambda`` (rPDC); ``threshold`` is the value of ``observed`` that a cell must
        exceed to be significant at ``alpha`` without correction. Undefined cells (the diagonal)
        are NaN.
    df : ndarray, shape (n_freqs,)
        Degrees of freedom of the chi-square distribution at every frequency.
    n_obs : int
        ``N``, the regression rows of the fit (the predicted samples of all trials).
    null : str
        The null hypothesis.
    limitations : tuple of str
        Ways the p-values can be wrong or misleading.
    alpha : float
        Level of ``threshold`` and default level for :meth:`significant`.
    """

    observed: Connectivity
    pvalues: np.ndarray
    statistic: np.ndarray
    threshold: np.ndarray
    df: np.ndarray
    n_obs: int
    null: str
    limitations: tuple[str, ...]
    alpha: float

    def __post_init__(self) -> None:
        """Make the arrays read-only."""
        for name in ("pvalues", "statistic", "threshold", "df"):
            view = np.asarray(getattr(self, name), dtype=float).view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def method(self) -> str:
        """Name of the estimator."""
        return self.observed.provenance.method

    @property
    def defined(self) -> np.ndarray:
        """Boolean mask of the cells that have a p-value."""
        return ~np.isnan(self.statistic)

    def __repr__(self) -> str:
        """Return a summary that includes the null hypothesis."""
        shape = "x".join(str(n) for n in self.pvalues.shape)
        return (
            f"AsymptoticTestResult(method={self.method!r}, shape={shape}, n_obs={self.n_obs}, "
            f"null={self.null!r})"
        )


def _refit(estimator: _SpectralVAR, data: TimeSeries) -> tuple[Connectivity, VarFit]:
    """Fit the estimator, then the VAR model behind its values once more at the recorded order.

    The second fit repeats the estimator's own steps (its preprocessing, the order it recorded,
    its demeaning) and so gives the same model; no order is selected again.
    """
    if estimator.ridge != 0.0:
        raise ValueError(
            "The asymptotic tests need ordinary least squares: use ridge=0 (with a penalty the "
            "coefficients have another covariance)."
        )
    observed = estimator.fit(data)
    series, _ = estimator._preprocess(data)
    order = int(observed.provenance.model_order or 0)
    return observed, fit_var(series, order, demean=estimator.demean)


def _with_nan_diagonal(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=float)
    n = out.shape[0]
    out[np.arange(n), np.arange(n)] = np.nan
    return out[..., None]


def analytic_pdc_test(
    estimator: PDC, data: TimeSeries, *, alpha: float = 0.05
) -> AsymptoticTestResult:
    """Test PDC at every frequency with its asymptotic distribution (Schelter et al. 2006).

    In the paper's indices (``i`` target, ``j`` source) the squared PDC ``|pi_ij(f)|^2`` is
    significant at level ``alpha`` when::

        |pi_ij(f)|^2 > C_ij(f) chi2_{1, 1-alpha} / (N sum_k |Abar_kj(f)|^2),

    with ``C_ij(f)`` from :func:`dirconn.spectral.pdc_null_scale`. The statistic is
    ``N |Abar_ij(f)|^2 / C_ij(f)``; it does not depend on the scale of the channels, so the
    p-values are the same with and without ``standardize``, while the threshold on the PDC values
    is not. For ``squared=False`` the threshold is the square root; the p-values are unchanged.

    Parameters
    ----------
    estimator : PDC
        Must have ``ridge=0``. Its frequency grid is the tested family; give it the ``freqs`` or
        ``fmax`` of a band to test only that band.
    data : TimeSeries
        Input data.
    alpha : float
        Level of the threshold and default level for :meth:`AsymptoticTestResult.significant`.

    Returns
    -------
    AsymptoticTestResult

    Raises
    ------
    TypeError
        If ``estimator`` is not a :class:`~dirconn.PDC` (gPDC included).
    ValueError
        With a ridge penalty.
    """
    if not isinstance(estimator, PDC):
        raise TypeError(
            "analytic_pdc_test needs a PDC estimator (gPDC has no asymptotic test here); got "
            f"{type(estimator).__name__}. Use surrogate_test or bootstrap for other methods."
        )
    alpha = check_alpha(alpha)
    observed, fit = _refit(estimator, data)
    freqs = observed.freqs
    scale = pdc_null_scale(fit.coefs, fit.noise_cov, fit.regressor_cov, freqs, fit.sfreq)
    power = (np.abs(ar_polynomial(fit.coefs, freqs, fit.sfreq)) ** 2).transpose(2, 1, 0)
    outflow = power.sum(axis=1, keepdims=True)  # [source, 1, freq], the PDC denominator
    statistic = fit.n_obs * power / scale
    threshold = scale * stats.chi2.ppf(1.0 - alpha, 1) / (fit.n_obs * outflow)
    if not estimator.squared:
        threshold = np.sqrt(threshold)
    statistic = _with_nan_diagonal(statistic)
    return AsymptoticTestResult(
        observed=observed,
        pvalues=np.where(np.isnan(statistic), np.nan, stats.chi2.sf(statistic, 1)),
        statistic=statistic,
        threshold=_with_nan_diagonal(threshold),
        df=np.ones(freqs.shape[0]),
        n_obs=fit.n_obs,
        null=SPECTRAL_NULL,
        limitations=PDC_LIMITATIONS,
        alpha=alpha,
    )


def analytic_rpdc_test(
    estimator: RPDC, data: TimeSeries, *, alpha: float = 0.05
) -> AsymptoticTestResult:
    """Test rPDC at every frequency with its asymptotic distribution (Schelter et al. 2009).

    ``N lambda_ij(f)`` is the Wald statistic of ``Abar_ij(f) = 0``: chi-square with ``df = 2``,
    or ``df = 1`` where the covariance of the real and imaginary part has rank 1 (at order 1, and
    at ``f = 0`` and the Nyquist frequency; :func:`dirconn.spectral.rank_one_freqs`). A value
    ``lambda`` is significant at level ``alpha`` when it exceeds ``chi2_{df, 1-alpha} / N``.

    Parameters
    ----------
    estimator : RPDC
        Its frequency grid is the tested family; give it the ``freqs`` or ``fmax`` of a band to
        test only that band.
    data : TimeSeries
        Input data.
    alpha : float
        Level of the threshold and default level for :meth:`AsymptoticTestResult.significant`.

    Returns
    -------
    AsymptoticTestResult

    Raises
    ------
    TypeError
        If ``estimator`` is not an :class:`~dirconn.RPDC`.
    """
    if not isinstance(estimator, RPDC):
        raise TypeError(
            f"analytic_rpdc_test needs an RPDC estimator; got {type(estimator).__name__}. Use "
            "analytic_pdc_test for PDC, surrogate_test or bootstrap for other methods."
        )
    alpha = check_alpha(alpha)
    observed, fit = _refit(estimator, data)
    freqs = observed.freqs
    values, df = rpdc_from_coefs(fit.coefs, fit.noise_cov, fit.regressor_cov, freqs, fit.sfreq)
    statistic = _with_nan_diagonal(fit.n_obs * values)
    threshold = np.broadcast_to(stats.chi2.ppf(1.0 - alpha, df) / fit.n_obs, values.shape)
    return AsymptoticTestResult(
        observed=observed,
        pvalues=np.where(
            np.isnan(statistic), np.nan, stats.chi2.sf(statistic, df[None, None, :, None])
        ),
        statistic=statistic,
        threshold=_with_nan_diagonal(threshold),
        df=df.astype(float),
        n_obs=fit.n_obs,
        null=SPECTRAL_NULL,
        limitations=SPECTRAL_LIMITATIONS,
        alpha=alpha,
    )
