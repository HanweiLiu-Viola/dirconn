import dataclasses
import warnings

import numpy as np
import pytest

from dirconn import AssumptionCheck, Connectivity, MethodInfo, TimeSeries
from dirconn.estimator import Estimator
from dirconn.exceptions import DirconnWarning
from dirconn.provenance import STANDARD_ASSUMPTIONS, array_hash


def make_info(**overrides):
    kwargs = dict(name="dummy", backend="test backend", directed=True)
    kwargs.update(overrides)
    return MethodInfo(**kwargs)


@dataclasses.dataclass(frozen=True)
class Dummy(Estimator):
    """Minimal estimator: every off-diagonal entry is 1."""

    info = make_info()
    order: int = 2
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.order < 1:
            raise ValueError("order must be at least 1")

    def _fit(self, data):
        n = data.n_channels
        values = np.ones((n, n))
        np.fill_diagonal(values, np.nan)
        provenance = self._provenance(data, model_order=self.order, seed=self.seed)
        return Connectivity.from_array(values, source=data.names, provenance=provenance)


@dataclasses.dataclass(frozen=True)
class ScaleDependent(Dummy):
    info = make_info(name="scaled", scale_dependent=True)


@dataclasses.dataclass(frozen=True)
class Pair(Dummy):
    info = make_info(name="pair", min_nodes=2, max_nodes=2)


@dataclasses.dataclass(frozen=True)
class WithExtraChecks(Dummy):
    info = make_info(name="extra")

    def _extra_checks(self, data):
        return [AssumptionCheck("stationarity", "warn", detail="drift", value=0.3)]


def make_ts(n_channels=3, n_times=200, scales=None, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_channels, n_times))
    if scales is not None:
        data = data * np.asarray(scales, dtype=float)[:, None]
    return TimeSeries.from_array(data, sfreq=100.0)


class TestMethodInfo:
    def test_defaults(self):
        info = make_info()
        assert info.min_nodes == 2
        assert info.max_nodes is None
        assert info.scale_dependent is False
        assert info.complex_valued is False
        assert info.frequency_resolved is False
        assert info.time_resolved is False
        assert info.value_range is None
        assert info.assumptions == ()

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_info().name = "x"  # type: ignore[misc]

    @pytest.mark.parametrize("field", ["name", "backend"])
    def test_name_and_backend_must_be_non_empty(self, field):
        with pytest.raises(ValueError, match=field):
            make_info(**{field: " "})

    def test_at_least_two_nodes_are_needed_for_a_directed_measure(self):
        with pytest.raises(ValueError, match="min_nodes"):
            make_info(min_nodes=1)

    def test_max_nodes_cannot_be_below_min_nodes(self):
        with pytest.raises(ValueError, match="max_nodes"):
            make_info(min_nodes=3, max_nodes=2)

    def test_value_range_must_be_ordered(self):
        assert make_info(value_range=(0.0, float("inf"))).value_range == (0.0, float("inf"))
        with pytest.raises(ValueError, match="value_range"):
            make_info(value_range=(1.0, 0.0))

    def test_assumptions_must_be_standard_names(self):
        info = make_info(assumptions=("stationarity", "linearity"))
        assert info.assumptions == ("stationarity", "linearity")
        with pytest.raises(ValueError, match="assumptions"):
            make_info(assumptions=("vibes",))

    def test_standard_assumptions_are_importable_for_reference(self):
        assert "stationarity" in STANDARD_ASSUMPTIONS


class TestEstimatorDefinition:
    def test_unknown_keyword_arguments_raise_type_error(self):
        with pytest.raises(TypeError):
            Dummy(ordr=3)  # type: ignore[call-arg]

    def test_parameters_are_typed_keywords_with_defaults(self):
        assert Dummy().order == 2
        assert Dummy(order=5, seed=1).seed == 1

    def test_get_params_returns_the_constructor_parameters_only(self):
        assert Dummy(order=4).get_params() == {"order": 4, "seed": None}

    def test_repr_lists_parameters(self):
        assert repr(Dummy(order=4)) == "Dummy(order=4, seed=None)"

    def test_estimators_are_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Dummy().order = 3  # type: ignore[misc]

    def test_parameter_validation_still_runs_through_the_base_post_init(self):
        with pytest.raises(ValueError, match="order"):
            Dummy(order=0)

    def test_missing_info_is_rejected_at_construction(self):
        @dataclasses.dataclass(frozen=True)
        class NoInfo(Estimator):
            def _fit(self, data):
                raise AssertionError

        with pytest.raises(TypeError, match="info"):
            NoInfo()

    def test_a_non_method_info_is_rejected(self):
        @dataclasses.dataclass(frozen=True)
        class BadInfo(Estimator):
            info = "granger"

            def _fit(self, data):
                raise AssertionError

        with pytest.raises(TypeError, match="info"):
            BadInfo()

    def test_estimator_without_fit_implementation_cannot_be_instantiated(self):
        @dataclasses.dataclass(frozen=True)
        class NoFit(Estimator):
            info = make_info()

        with pytest.raises(TypeError):
            NoFit()

    def test_non_dataclass_estimator_is_rejected_when_fitting(self):
        class Plain(Estimator):
            info = make_info()

            def _fit(self, data):
                raise AssertionError

        with pytest.raises(TypeError, match="frozen dataclass"):
            Plain().fit(make_ts())

    def test_non_frozen_dataclass_is_rejected(self):
        @dataclasses.dataclass
        class Mutable(Estimator):
            info = make_info()

            def _fit(self, data):
                raise AssertionError

        with pytest.raises(TypeError, match="frozen"):
            Mutable()


class TestFit:
    def test_fit_returns_a_connectivity_and_keeps_the_estimator_stateless(self):
        est = Dummy()
        ts = make_ts()
        result = est.fit(ts)
        assert isinstance(result, Connectivity)
        assert result.source == ts.names
        assert est.get_params() == {"order": 2, "seed": None}

    def test_fit_can_be_repeated_on_different_data(self):
        est = Dummy()
        assert est.fit(make_ts(3)).values.shape[0] == 3
        assert est.fit(make_ts(4)).values.shape[0] == 4

    def test_provenance_describes_the_estimator_and_the_input(self):
        ts = make_ts()
        prov = Dummy(order=3, seed=7).fit(ts).provenance
        assert prov.method == "dummy"
        assert prov.backend == "test backend"
        assert prov.parameters == {"order": 3, "seed": 7}
        assert prov.model_order == 3
        assert prov.seed == 7
        assert prov.input_hash == array_hash(ts.data)
        assert prov.claim == "predictive, not interventional"

    def test_a_non_time_series_input_is_rejected(self):
        with pytest.raises(TypeError, match="TimeSeries"):
            Dummy().fit(np.zeros((3, 100)))  # type: ignore[arg-type]

    def test_fit_must_produce_a_connectivity(self):
        @dataclasses.dataclass(frozen=True)
        class Wrong(Estimator):
            info = make_info()

            def _fit(self, data):
                return np.zeros((2, 2))

        with pytest.raises(TypeError, match="Connectivity"):
            Wrong().fit(make_ts())

    def test_unserialisable_parameters_are_caught_when_the_provenance_is_built(self):
        @dataclasses.dataclass(frozen=True)
        class Callback(Dummy):
            hook: object = None

        with pytest.raises(TypeError, match="JSON"):
            Callback(hook=print).fit(make_ts())


class TestNodeCountGuard:
    def test_too_few_channels_are_rejected(self):
        with pytest.raises(ValueError, match="at least 2 channels"):
            Dummy().fit(make_ts(n_channels=1))

    def test_too_many_channels_are_rejected(self):
        with pytest.raises(ValueError, match="at most 2 channels"):
            Pair().fit(make_ts(n_channels=3))

    def test_the_allowed_count_passes(self):
        assert Pair().fit(make_ts(n_channels=2)).values.shape[:2] == (2, 2)

    def test_the_error_names_the_method(self):
        with pytest.raises(ValueError, match="pair"):
            Pair().fit(make_ts(n_channels=3))


class TestScaleGuard:
    def test_unbalanced_channel_scales_warn_and_are_recorded(self):
        ts = make_ts(scales=[1.0, 100.0, 1.0])
        with pytest.warns(DirconnWarning, match="standardi"):
            result = ScaleDependent().fit(ts)
        check = result.provenance.assumptions.get("channel_scale")
        assert check.status == "warn"
        assert check.value == pytest.approx(100.0, rel=0.2)
        assert result.provenance.assumptions.has_warnings

    def test_balanced_channels_pass_silently(self):
        ts = make_ts(scales=[1.0, 2.0, 1.5])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = ScaleDependent().fit(ts)
        assert result.provenance.assumptions.get("channel_scale").status == "pass"

    def test_a_flat_channel_warns_without_a_numeric_value(self):
        data = np.random.default_rng(0).standard_normal((3, 100))
        data[1] = 0.0
        ts = TimeSeries.from_array(data, sfreq=100.0)
        with pytest.warns(DirconnWarning):
            result = ScaleDependent().fit(ts)
        check = result.provenance.assumptions.get("channel_scale")
        assert check.status == "warn"
        assert check.value is None

    def test_scale_independent_methods_do_not_check_the_scale(self):
        ts = make_ts(scales=[1.0, 1000.0, 1.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = Dummy().fit(ts)
        with pytest.raises(KeyError):
            result.provenance.assumptions.get("channel_scale")

    def test_the_standard_assumptions_stay_in_the_checklist(self):
        ts = make_ts(scales=[1.0, 2.0, 1.5])
        names = {c.name for c in ScaleDependent().fit(ts).provenance.assumptions.checks}
        assert set(STANDARD_ASSUMPTIONS) <= names


class TestExtraChecksHook:
    def test_checks_from_the_hook_are_merged_into_the_result(self):
        result = WithExtraChecks().fit(make_ts())
        check = result.provenance.assumptions.get("stationarity")
        assert check.status == "warn"
        assert check.value == 0.3
        assert result.provenance.assumptions.get("linearity").status == "not_checked"

    def test_the_default_hook_adds_nothing(self):
        result = Dummy().fit(make_ts())
        assert {c.status for c in result.provenance.assumptions.checks} == {"not_checked"}
