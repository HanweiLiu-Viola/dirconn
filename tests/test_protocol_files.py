"""The shipped protocol files (protocols/*.yaml) are valid and mean what they say."""

from pathlib import Path

import pytest

from dirconn.protocol import VAR_ESTIMATOR_NAMES, expand, load_protocol
from dirconn.report import ADDONS

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "protocols"
FILES = sorted(PROTOCOL_DIR.glob("*.yaml"))


def test_there_are_protocol_files():
    assert {"quick"} <= {p.stem for p in FILES}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
class TestEveryFile:
    def test_it_loads_and_is_named_after_its_file(self, path):
        assert load_protocol(path).name == path.stem

    def test_it_has_a_description(self, path):
        assert load_protocol(path).description

    def test_it_expands_to_tasks(self, path):
        assert expand(load_protocol(path))


class TestFull:
    def test_it_has_the_saturation_guard_and_the_grid_that_can_pass_it(self):
        protocol = load_protocol(PROTOCOL_DIR / "full.yaml")
        assert protocol.saturation_guard is True
        assert "chain" in protocol.scenarios  # the reference scenario of the guard
        # a hard end (a probe: the chain breaks from -10 dB with 30 s)
        # and an easy anchor
        assert min(protocol.snr_db) <= -10 and max(protocol.snr_db) >= 0
        assert min(protocol.duration_s) <= 30

    def test_the_other_protocols_do_not_ask_for_the_guard(self):
        for path in FILES:
            if path.stem != "full":
                assert load_protocol(path).saturation_guard is False, path.stem


class TestQuick:
    def test_it_is_the_planned_24_task_grid(self):
        protocol = load_protocol(PROTOCOL_DIR / "quick.yaml")
        tasks = expand(protocol)
        assert (
            len(tasks) == 3 * 2 * 1 * 1 * 2 * 2
        )  # scenarios x snr x length x source x modality x seeds
        assert {t.modality for t in tasks} == {"eeg", "grad"}
        assert protocol.n_seeds == 2

    def test_it_has_every_estimator_at_both_order_settings(self):
        assert set(load_protocol(PROTOCOL_DIR / "quick.yaml").estimators) == set(
            VAR_ESTIMATOR_NAMES
        )

    def test_its_budget_is_minutes(self):
        assert load_protocol(PROTOCOL_DIR / "quick.yaml").budget_hours <= 0.25


class TestBudgetShares:
    """The two hours of the plan are for all protocols together, not for each one."""

    def test_the_budgets_of_the_published_protocols_fit_into_two_hours(self):
        # the add-ons have a budget of their own (below), the quick tier is not counted
        shares = {
            p.stem: load_protocol(p).budget_hours
            for p in FILES
            if p.stem != "quick" and p.stem not in ADDONS
        }
        # shares are the gate estimates rounded up with a little room (docs/benchmark.md)
        assert sum(shares.values()) <= 2.0 + 1e-9, shares  # the gate estimates sum to about 1.83 h
        assert "full" in shares and shares["full"] >= max(shares.values())

    def test_the_two_leakage_protocols_share_the_leakage_budget(self):
        assert load_protocol(PROTOCOL_DIR / "leakage_adjacent.yaml").budget_hours == 0.04
        assert load_protocol(PROTOCOL_DIR / "leakage_pairs.yaml").budget_hours == 0.06


class TestLeakageProtocols:
    def test_the_neighbour_protocol_keeps_the_scores_and_covers_both_modalities_and_snrs(self):
        protocol = load_protocol(PROTOCOL_DIR / "leakage_adjacent.yaml")
        assert protocol.scenarios == ("leakage_adjacent",)
        assert protocol.settings == {"store_scores": True}
        # one estimator of each kind, BIC order only: the gate estimated 0.15 h for all ten labels
        assert set(protocol.estimators) == {
            "gc/bic",
            "gc_pairwise/bic",
            "dtf/bic",
            "pdc/bic",
            "gpdc/bic",
        }
        tasks = expand(protocol)
        assert len(tasks) == 2 * 1 * 10  # modalities x SNRs x seeds
        assert {t.modality for t in tasks} == {"eeg", "grad"}
        assert {t.snr_db for t in tasks} == {10.0}
        assert protocol.n_seeds == 10  # the plan's ten seeds are not given up for the budget

    def test_the_pairs_protocol_is_45_pairs_on_two_modalities(self):
        protocol = load_protocol(PROTOCOL_DIR / "leakage_pairs.yaml")
        assert protocol.kind == "region_pairs" and protocol.settings == {"store_scores": True}
        assert protocol.pairs is not None and protocol.pairs["n"] == 45
        assert set(protocol.estimators) == {"gc/bic", "gpdc/bic"}
        tasks = expand(protocol)
        assert len(tasks) == 45 * 2
        assert len({t.scenario for t in tasks}) == 45
        assert all(t.zero_network and t.n_nodes == 2 for t in tasks)

    def test_the_regions_are_unique_labels_and_enough_for_the_pairs(self):
        protocol = load_protocol(PROTOCOL_DIR / "leakage_pairs.yaml")
        regions = protocol.pairs["regions"]
        assert len(set(regions)) == len(regions) >= 20
        assert all(r.endswith(("-lh", "-rh")) for r in regions)

    def test_neither_asks_for_the_saturation_guard(self):
        for name in ("leakage_adjacent", "leakage_pairs"):
            assert load_protocol(PROTOCOL_DIR / f"{name}.yaml").saturation_guard is False


class TestSignificanceProtocol:
    def protocol(self):
        return load_protocol(PROTOCOL_DIR / "significance.yaml")

    def test_it_is_four_worlds_of_the_chain_with_25_datasets_each(self):
        protocol = self.protocol()
        assert protocol.kind == "significance" and protocol.scenarios == ("chain",)
        assert protocol.coupling == (0.0, 0.4) and protocol.snr_db == (10.0, 40.0)
        tasks = expand(protocol)
        assert len(tasks) == 4 * 25
        assert {t.modality for t in tasks} == {"roi"}
        assert sum(t.zero_network for t in tasks) == 50

    def test_it_has_both_kinds_of_test_and_the_39_surrogates_of_the_calibration(self):
        protocol = self.protocol()
        assert protocol.tests == ("analytic", "surrogate:phase_randomize")
        assert protocol.settings == {"n_surrogates": 39}
        assert protocol.estimators == ("gc/fixed",)

    def test_its_share_is_six_hundredths_of_an_hour(self):
        assert self.protocol().budget_hours == 0.06
        assert self.protocol().saturation_guard is False


class TestNoiseProtocol:
    def protocol(self):
        return load_protocol(PROTOCOL_DIR / "noise.yaml")

    def test_it_is_three_arms_on_both_kinds_of_meg_sensor(self):
        protocol = self.protocol()
        assert protocol.noise == ("simulated", "white", "empty_room")
        assert protocol.modality == ("mag", "grad")
        tasks = expand(protocol)
        assert len(tasks) == 3 * 2 * 9  # arms x modalities x seeds
        assert {t.noise for t in tasks} == {"simulated", "white", "empty_room"}

    def test_the_seeds_fill_the_three_windows_of_the_recording_evenly(self):
        # 30 s blocks of the 110 s recording: three windows, used by seed index modulo 3
        assert self.protocol().n_seeds % 3 == 0
        assert self.protocol().duration_s == (30.0,)

    def test_the_snr_is_the_one_where_the_chain_stops_being_perfect(self):
        assert self.protocol().snr_db == (-10.0,)

    def test_it_has_its_five_hundredths_of_an_hour_and_no_guard(self):
        assert self.protocol().budget_hours == 0.05
        assert self.protocol().saturation_guard is False


class TestAddOnProtocols:
    """Noise at harder SNR and the significance study with more power, run after the ten."""

    def protocol(self, name):
        return load_protocol(PROTOCOL_DIR / f"{name}.yaml")

    def test_every_add_on_has_a_file_and_a_budget_of_at_most_a_tenth_of_an_hour(self):
        for name in ADDONS:
            assert (PROTOCOL_DIR / f"{name}.yaml").is_file(), name
            # at most 0.1 h each; m5_methods 0.3 h, the calibration the gate estimates of its design
            limit = {"m5_methods": 0.3, "asymptotic_calibration": 1.5}.get(name, 0.1)
            assert 0 < self.protocol(name).budget_hours <= limit, name
            assert self.protocol(name).saturation_guard is False, name

    def test_noise_hard_is_noise_at_minus_15_and_minus_20_db_and_nothing_else_changed(self):
        hard, base = self.protocol("noise_hard"), self.protocol("noise")
        assert hard.snr_db == (-15.0, -20.0)
        for field in ("scenarios", "noise", "estimators", "n_seeds", "duration_s", "source_model"):
            assert getattr(hard, field) == getattr(base, field), field
        assert hard.modality == base.modality == ("mag", "grad")
        assert len(expand(hard)) == 2 * len(expand(base))  # two SNRs where noise has one

    def test_noise_hard_has_none_of_the_task_ids_of_noise(self):
        # a different SNR is a different task, so the add-on never repeats a published task
        ids = {t.task_id for t in expand(self.protocol("noise"))}
        assert not ids & {t.task_id for t in expand(self.protocol("noise_hard"))}

    def test_significance_power_adds_weak_couplings_datasets_and_surrogates(self):
        power, base = self.protocol("significance_power"), self.protocol("significance")
        assert power.kind == "significance" and power.scenarios == ("chain",)
        assert power.coupling == (0.0, 0.1, 0.2, 0.4) and power.snr_db == base.snr_db
        assert power.n_seeds == 40 and power.settings == {"n_surrogates": 99}
        assert power.tests == base.tests and power.duration_s == base.duration_s
        assert len(expand(power)) == 4 * 2 * 40  # couplings x SNRs x datasets

    def test_significance_power_keeps_the_fixed_order_of_the_first_study(self):
        # the BIC arm was dropped to fit the budget: the order setting is not varied here
        power, base = self.protocol("significance_power"), self.protocol("significance")
        assert power.estimators == base.estimators == ("gc/fixed",)

    def test_99_surrogates_let_two_true_edges_pass_benjamini_hochberg_over_six_cells(self):
        # the smallest p-value is 1/(n+1); the k-th smallest has to be <= k * q / m
        n_surrogates = self.protocol("significance_power").settings["n_surrogates"]
        smallest_p, q, m, true_edges = 1 / (n_surrogates + 1), 0.05, 6, 2
        assert smallest_p <= true_edges * q / m  # 0.01 <= 0.0167
        base = self.protocol("significance").settings["n_surrogates"]
        assert 1 / (base + 1) > true_edges * q / m  # 0.025: why the first study had power 0


@pytest.mark.sample_data
def test_the_pairs_protocol_is_what_the_script_writes():
    import importlib.util

    import yaml

    path = PROTOCOL_DIR.parent / "scripts" / "make_leakage_pairs_protocol.py"
    spec = importlib.util.spec_from_file_location("make_leakage_pairs_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shipped = yaml.safe_load((PROTOCOL_DIR / "leakage_pairs.yaml").read_text(encoding="utf-8"))
    regions = module.candidate_regions(module.MIN_VISIBILITY)
    assert shipped == module.build(regions)


class TestSlices:
    """Coupling, extent and size: the axes that `full` leaves out, as protocols of their own."""

    def estimators(self, name):
        return set(load_protocol(PROTOCOL_DIR / f"{name}.yaml").estimators)

    def test_the_three_slices_have_the_five_estimators_at_bic_order(self):
        five = {"gc/bic", "gc_pairwise/bic", "dtf/bic", "pdc/bic", "gpdc/bic"}
        for name in ("coupling", "extent", "size"):
            assert self.estimators(name) == five, name

    def test_coupling_is_three_simple_scenarios_at_two_strengths(self):
        protocol = load_protocol(PROTOCOL_DIR / "coupling.yaml")
        assert protocol.scenarios == ("chain", "collider", "common_driver")
        assert protocol.coupling == (0.1, 0.2)
        assert (
            len(expand(protocol)) == 3 * 2 * 3 * 2 * 10
        )  # scenarios x couplings x SNR x length x seeds

    def test_extent_is_the_focal_and_the_patch_source(self):
        protocol = load_protocol(PROTOCOL_DIR / "extent.yaml")
        assert protocol.source_model == ("focal", "patch") and protocol.scenarios == (
            "chain",
            "collider",
        )
        assert len(expand(protocol)) == 2 * 2 * 3 * 2 * 10

    def test_size_is_the_ten_region_graph(self):
        protocol = load_protocol(PROTOCOL_DIR / "size.yaml")
        assert protocol.scenarios == ("random_10",)
        assert {t.n_nodes for t in expand(protocol)} == {10}
        assert len(expand(protocol)) == 3 * 2 * 10

    def test_every_slice_has_the_grid_of_full_so_that_it_can_be_read_against_it(self):
        full = load_protocol(PROTOCOL_DIR / "full.yaml")
        for name in ("coupling", "extent", "size"):
            protocol = load_protocol(PROTOCOL_DIR / f"{name}.yaml")
            assert protocol.snr_db == full.snr_db and protocol.duration_s == full.duration_s, name
            assert protocol.modality == full.modality and protocol.n_seeds == full.n_seeds, name


class TestCalibrationProtocol:
    """The calibration of the asymptotic PDC and rPDC tests."""

    def protocol(self):
        return load_protocol(PROTOCOL_DIR / "asymptotic_calibration.yaml")

    def test_it_is_the_design_of_decision_15(self):
        protocol = self.protocol()
        assert protocol.kind == "significance" and protocol.scenarios == ("chain",)
        assert protocol.coupling == (0.0, 0.4) and protocol.snr_db == (10.0, 40.0)
        assert protocol.duration_s == (30.0, 120.0) and protocol.n_seeds == 25
        assert len(expand(protocol)) == 2 * 2 * 2 * 25

    def test_pdc_and_rpdc_at_the_judged_order_a_too_low_one_and_bic(self):
        protocol = self.protocol()
        assert set(protocol.estimators) == {
            f"{m}/{o}" for m in ("pdc", "rpdc") for o in ("fixed", "low", "bic")
        }
        assert protocol.settings["low_order"] == 2

    def test_the_asymptotic_tests_next_to_39_surrogates_on_a_grid_of_one_hertz(self):
        protocol = self.protocol()
        assert protocol.tests == ("analytic", "surrogate:phase_randomize")
        assert protocol.settings["n_surrogates"] == 39
        assert protocol.settings["n_freqs"] == 51  # 0 to 50 Hz at 100 Hz sampling
