import dataclasses
import json

import numpy as np
import pytest

import dirconn
from dirconn import AssumptionCheck, AssumptionChecklist, Provenance
from dirconn.provenance import STANDARD_ASSUMPTIONS, array_hash


class TestAssumptionChecklist:
    def test_unchecked_lists_every_standard_assumption(self):
        checklist = AssumptionChecklist.unchecked()
        assert tuple(c.name for c in checklist.checks) == STANDARD_ASSUMPTIONS
        assert {c.status for c in checklist.checks} == {"not_checked"}

    def test_standard_assumptions_cover_the_spec(self):
        assert set(STANDARD_ASSUMPTIONS) == {
            "stationarity",
            "linearity",
            "sampling_adequacy",
            "instantaneous_coupling",
            "causal_sufficiency",
            "filtering",
        }

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValueError, match="status"):
            AssumptionCheck("stationarity", "maybe")

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            AssumptionCheck("", "pass")

    def test_with_check_replaces_an_existing_entry_without_mutating(self):
        original = AssumptionChecklist.unchecked()
        updated = original.with_check(
            AssumptionCheck("stationarity", "warn", detail="ADF p=0.2", value=0.2)
        )
        assert original.get("stationarity").status == "not_checked"
        assert updated.get("stationarity").status == "warn"
        assert updated.get("stationarity").value == 0.2
        assert len(updated.checks) == len(original.checks)

    def test_with_check_appends_a_new_name(self):
        updated = AssumptionChecklist.unchecked().with_check(AssumptionCheck("custom", "pass"))
        assert updated.get("custom").status == "pass"
        assert len(updated.checks) == len(STANDARD_ASSUMPTIONS) + 1

    def test_get_unknown_name_raises(self):
        with pytest.raises(KeyError):
            AssumptionChecklist.unchecked().get("nope")

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            AssumptionChecklist((AssumptionCheck("a", "pass"), AssumptionCheck("a", "warn")))

    def test_has_warnings(self):
        checklist = AssumptionChecklist.unchecked()
        assert not checklist.has_warnings
        assert checklist.with_check(AssumptionCheck("linearity", "warn")).has_warnings
        assert checklist.with_check(AssumptionCheck("linearity", "fail")).has_warnings

    def test_list_round_trip(self):
        checklist = AssumptionChecklist.unchecked().with_check(
            AssumptionCheck("filtering", "warn", detail="band-pass 1-45 Hz", value=None)
        )
        assert AssumptionChecklist.from_list(checklist.to_list()) == checklist


class TestProvenance:
    def make(self, **overrides):
        kwargs = dict(method="granger", parameters={"order": 5})
        kwargs.update(overrides)
        return Provenance(**kwargs)

    def test_defaults_state_the_claim_and_carry_an_unchecked_checklist(self):
        prov = self.make()
        assert prov.claim == "predictive, not interventional"
        assert prov.assumptions == AssumptionChecklist.unchecked()
        assert prov.history == ()
        assert prov.model_order is None
        assert prov.seed is None
        assert prov.input_hash is None

    def test_versions_are_collected(self):
        versions = self.make().versions
        assert versions["dirconn"] == dirconn.__version__
        assert {"python", "numpy", "scipy", "pandas", "xarray"} <= set(versions)

    def test_method_is_required_and_non_empty(self):
        with pytest.raises(ValueError, match="method"):
            Provenance(method="  ")

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.make().method = "pdc"  # type: ignore[misc]

    def test_parameters_are_isolated_from_the_callers_dict(self):
        params = {"order": 5, "bands": [1, 4]}
        prov = self.make(parameters=params)
        params["order"] = 99
        params["bands"].append(8)
        assert prov.parameters == {"order": 5, "bands": [1, 4]}

    def test_numpy_values_and_tuples_are_normalised_to_json_types(self):
        prov = self.make(
            parameters={
                "alpha": np.float64(0.05),
                "n": np.int64(3),
                "grid": np.array([1.0, 2.0]),
                "band": (8, 13),
                "flag": np.bool_(True),
            },
            model_order=np.int64(4),
            seed=np.int64(7),
        )
        assert prov.parameters == {
            "alpha": 0.05,
            "n": 3,
            "grid": [1.0, 2.0],
            "band": [8, 13],
            "flag": True,
        }
        assert type(prov.model_order) is int
        assert type(prov.seed) is int

    def test_unserialisable_parameter_is_rejected(self):
        with pytest.raises(TypeError, match="JSON"):
            self.make(parameters={"fn": object()})

    def test_nan_parameter_is_rejected_to_keep_the_json_strict(self):
        with pytest.raises(ValueError, match="finite"):
            self.make(parameters={"alpha": float("nan")})

    def test_backend_defaults_to_none_and_is_kept(self):
        assert self.make().backend is None
        assert self.make(backend="own numpy OLS").backend == "own numpy OLS"

    def test_json_round_trip(self):
        prov = self.make(
            model_order=5,
            seed=0,
            backend="own numpy OLS",
            input_hash="abc",
            preprocessing={"filter": None, "reference": "average", "standardised": True},
            assumptions=AssumptionChecklist.unchecked().with_check(
                AssumptionCheck("stationarity", "pass", value=0.01)
            ),
        ).with_step({"op": "reduce_freq", "band": [8, 13], "how": "mean"})
        restored = Provenance.from_json(prov.to_json())
        assert restored == prov
        json.loads(prov.to_json())  # valid strict JSON

    def test_dict_round_trip(self):
        prov = self.make(seed=3)
        assert Provenance.from_dict(prov.to_dict()) == prov

    def test_details_default_to_empty_and_are_normalised_like_parameters(self):
        assert self.make().details == {}
        prov = self.make(details={"curve": np.array([0.5, 0.25]), "n_obs": np.int64(7)})
        assert prov.details == {"curve": [0.5, 0.25], "n_obs": 7}

    def test_details_survive_json_and_with_step(self):
        prov = self.make(details={"order_selection": {"orders": [1, 2], "cv": None}})
        assert Provenance.from_json(prov.to_json()) == prov
        assert prov.with_step({"op": "x"}).details == prov.details

    def test_old_dictionaries_without_details_still_load(self):
        data = self.make().to_dict()
        del data["details"]
        assert Provenance.from_dict(data).details == {}

    def test_with_step_appends_without_mutating(self):
        prov = self.make()
        stepped = prov.with_step({"op": "reduce_freq", "band": (8, 13)})
        assert prov.history == ()
        assert stepped.history == ({"op": "reduce_freq", "band": [8, 13]},)

    def test_from_dict_rejects_missing_method(self):
        with pytest.raises(ValueError, match="method"):
            Provenance.from_dict({"parameters": {}})


class TestArrayHash:
    def test_same_content_gives_same_hash(self):
        a = np.arange(12.0).reshape(3, 4)
        assert array_hash(a) == array_hash(a.copy())

    def test_hash_is_a_sha256_hex_digest(self):
        assert len(array_hash(np.zeros(3))) == 64

    def test_different_values_differ(self):
        a = np.arange(6.0)
        b = a.copy()
        b[2] += 1e-12
        assert array_hash(a) != array_hash(b)

    def test_shape_is_part_of_the_hash(self):
        a = np.arange(6.0)
        assert array_hash(a.reshape(2, 3)) != array_hash(a.reshape(3, 2))

    def test_dtype_is_part_of_the_hash(self):
        a = np.arange(6)
        assert array_hash(a.astype(np.float32)) != array_hash(a.astype(np.float64))

    def test_memory_layout_does_not_matter(self):
        a = np.arange(12.0).reshape(3, 4)
        assert array_hash(a.T) == array_hash(np.ascontiguousarray(a.T))


def test_collected_versions_are_a_private_copy_each_time():
    # The lookup is cached because it is slow; a caller changing its dict must not change the next.
    from dirconn.provenance import collect_versions

    first = collect_versions()
    first["python"] = "tampered"
    first["extra"] = "x"
    second = collect_versions()
    assert second["python"] != "tampered" and "extra" not in second


def test_inside_the_docker_image_the_image_is_recorded(monkeypatch):
    from dirconn import provenance

    monkeypatch.setenv("DIRCONN_IMAGE", "ghcr.io/x/dirconn:0.1-abcdef0")
    provenance._installed_versions.cache_clear()
    try:
        assert provenance.collect_versions()["image"] == "ghcr.io/x/dirconn:0.1-abcdef0"
    finally:
        monkeypatch.delenv("DIRCONN_IMAGE")
        provenance._installed_versions.cache_clear()
    assert "image" not in provenance.collect_versions()
