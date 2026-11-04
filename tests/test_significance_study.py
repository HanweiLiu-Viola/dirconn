"""Analysis of the significance study: rates, BH, power, minimum p-values."""

import math

import numpy as np
import pytest

from dirconn.protocol import render_summary
from dirconn.significance_study import frequency_report, significance_report

NAN = math.nan
NODES = ["transversetemporal-rh", "cuneus-rh", "transversetemporal-lh"]  # the chain
ENV = {"python": "3.11", "numpy": "2.4.6"}


def matrix(fill=0.5, **cells):
    """A 3 x 3 p-value matrix with a NaN diagonal; `cells` sets entries like c01=0.03."""
    out = [[fill] * 3 for _ in range(3)]
    for i in range(3):
        out[i][i] = NAN
    for key, value in cells.items():
        out[int(key[1])][int(key[2])] = value
    return out


def record(seed, coupling, tests, *, snr=10.0, estimator="gc/fixed", status="ok", scenario="chain"):
    """A task record; `tests` maps a test label to a 3 x 3 p-value matrix."""
    return {
        "type": "task",
        "status": status,
        "id": f"{scenario}-{coupling}-{snr}-{seed}-{estimator}",
        "task": {
            "scenario": scenario,
            "coupling": coupling,
            "snr_db": snr,
            "duration_s": 30.0,
            "modality": "roi",
            "source_model": "none",
            "seed_index": seed,
        },
        "result": {
            estimator: {
                "analysed": NODES,
                "tests": {label: {"pvalues": p} for label, p in tests.items()},
            }
        },
    }


ANALYTIC = "analytic"
SURROGATE = "surrogate:phase_randomize"


def only(rows, **where):
    (row,) = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return row


class TestNullWorld:
    def records(self):
        # 4 datasets x 6 cells = 24 null cells; p = 0.03 twice, 0.008 once, the rest 0.5
        both = lambda m: {ANALYTIC: m, SURROGATE: m}  # noqa: E731
        return [
            record(0, 0.0, both(matrix(c01=0.03))),
            record(1, 0.0, both(matrix(c10=0.03, c20=0.008))),
            record(2, 0.0, both(matrix())),
            record(3, 0.0, both(matrix())),
        ]

    def test_the_rejection_rate_of_the_null_cells_at_each_level(self):
        row = only(significance_report(self.records()), test=ANALYTIC)
        assert row["n_datasets"] == 4 and row["n_null_cells"] == 24 and row["n_edge_cells"] == 0
        assert row["null_rate"][0.05]["estimate"] == pytest.approx(3 / 24)
        assert row["null_rate"][0.10]["estimate"] == pytest.approx(3 / 24)
        assert row["null_rate"][0.01]["estimate"] == pytest.approx(1 / 24)
        assert row["power"][0.05] is None  # no edge to find in this world

    def test_a_level_below_the_smallest_p_value_of_a_surrogate_test_is_not_applicable(self):
        rows = significance_report(self.records(), n_surrogates=39)
        assert only(rows, test=SURROGATE)["null_rate"][0.01] is None  # 1 / 40 = 0.025 > 0.01
        assert only(rows, test=SURROGATE)["null_rate"][0.05] is not None
        assert only(rows, test=ANALYTIC)["null_rate"][0.01] is not None

    def test_the_smallest_p_value_follows_the_number_of_surrogates(self):
        rows = significance_report(self.records(), n_surrogates=199)  # 1 / 200 = 0.005
        assert only(rows, test=SURROGATE)["null_rate"][0.01] is not None

    def test_a_family_wise_rejection_after_bh_needs_a_p_value_small_enough_for_six_cells(self):
        records = [
            record(0, 0.0, {ANALYTIC: matrix(c01=0.001)}),  # 0.001 * 6 = 0.006 <= 0.05
            record(1, 0.0, {ANALYTIC: matrix(c01=0.03)}),  # 0.03 * 6 = 0.18 > 0.05
            record(2, 0.0, {ANALYTIC: matrix()}),
            record(3, 0.0, {ANALYTIC: matrix()}),
        ]
        row = only(significance_report(records), test=ANALYTIC)
        any_false = row["bh"]["any_false_rejection"]
        assert any_false["estimate"] == pytest.approx(0.25)
        assert 0 < any_false["low"] < 0.25 < any_false["high"] < 1

    def test_the_mean_p_value_is_reported_with_the_one_that_is_expected(self):
        rows = significance_report(self.records(), n_surrogates=39)
        analytic = only(rows, test=ANALYTIC)
        surrogate = only(rows, test=SURROGATE)
        assert analytic["expected_mean_p"] == pytest.approx(0.5)
        assert surrogate["expected_mean_p"] == pytest.approx(41 / 80)  # p = (k + 1) / 40
        expected = np.mean([0.5] * 21 + [0.03, 0.03, 0.008])
        assert analytic["mean_p_null"] == pytest.approx(expected)


class TestResolutionOfBh:
    """With few surrogates BH cannot reject a cell on its own: that is not a lack of power."""

    def records(self):
        return [record(0, 0.4, {ANALYTIC: matrix(), SURROGATE: matrix()})]

    def test_the_analytic_test_needs_one_cell_and_a_few_surrogates_need_several(self):
        rows = significance_report(self.records(), n_surrogates=39)
        assert only(rows, test=ANALYTIC)["bh_cells_needed"] == 1
        # p_min = 1 / 40 = 0.025; adjusted over 6 cells: 0.025 * 6 / k <= 0.05 needs k >= 3
        assert only(rows, test=SURROGATE)["bh_cells_needed"] == 3

    def test_more_surrogates_need_fewer_cells(self):
        assert (
            only(significance_report(self.records(), n_surrogates=59), test=SURROGATE)[
                "bh_cells_needed"
            ]
            == 2
        )
        assert (
            only(significance_report(self.records(), n_surrogates=119), test=SURROGATE)[
                "bh_cells_needed"
            ]
            == 1
        )

    def test_the_level_of_the_correction_matters(self):
        rows = significance_report(self.records(), n_surrogates=39, q=0.10)
        assert only(rows, test=SURROGATE)["bh_cells_needed"] == 2  # 0.15 / k <= 0.10

    def test_two_true_edges_at_the_smallest_p_value_are_not_enough_for_bh_with_39_surrogates(self):
        edges = dict(c01=1 / 40, c12=1 / 40)
        records = [record(s, 0.4, {SURROGATE: matrix(**edges)}) for s in range(5)]
        row = only(significance_report(records, n_surrogates=39), test=SURROGATE)
        assert row["power"][0.05]["estimate"] == pytest.approx(1.0)  # uncorrected: found
        assert row["bh"]["power"]["estimate"] == pytest.approx(0.0)  # corrected: 0.025 * 6 / 2


class TestAlternativeWorld:
    def records(self):
        # chain: the true edges are cells (0, 1) and (1, 2); the other four cells are null
        edges = dict(c01=0.0001, c12=0.0001)
        return [
            record(0, 0.4, {ANALYTIC: matrix(**edges)}),
            record(1, 0.4, {ANALYTIC: matrix(c20=0.02, **edges)}),
            record(2, 0.4, {ANALYTIC: matrix(**edges)}),
            record(3, 0.4, {ANALYTIC: matrix(c01=0.0001, c12=0.4)}),
        ]

    def test_power_is_over_the_true_edges_and_the_null_cells_are_counted_apart(self):
        row = only(significance_report(self.records()), test=ANALYTIC)
        assert row["n_edge_cells"] == 8 and row["n_null_cells"] == 16
        assert row["power"][0.05]["estimate"] == pytest.approx(7 / 8)
        assert row["null_rate"][0.05]["estimate"] == pytest.approx(1 / 16)

    def test_bh_power_and_the_false_discovery_proportion(self):
        row = only(significance_report(self.records()), test=ANALYTIC)
        assert row["bh"]["power"]["estimate"] == pytest.approx(7 / 8)
        # dataset 1 rejects the two edges and the null cell (p = 0.02, adjusted 0.04): 1 / 3
        assert row["bh"]["fdp"] == pytest.approx((0 + 1 / 3 + 0 + 0) / 4)
        assert row["bh"]["any_false_rejection"]["estimate"] == pytest.approx(0.25)

    def test_the_file_strengths_are_the_truth_when_the_coupling_is_not_set(self):
        records = [record(0, None, {ANALYTIC: matrix(c01=0.0001, c12=0.0001)})]
        row = only(significance_report(records), test=ANALYTIC)
        assert row["n_edge_cells"] == 2 and row["n_null_cells"] == 4


class TestIntervals:
    def test_identical_datasets_give_a_point_interval_and_different_ones_do_not(self):
        same = [record(s, 0.0, {ANALYTIC: matrix(c01=0.03)}) for s in range(10)]
        row = only(significance_report(same, seed=0), test=ANALYTIC)
        interval = row["null_rate"][0.05]
        assert interval["estimate"] == pytest.approx(1 / 6)
        assert interval["low"] == pytest.approx(1 / 6)
        assert interval["low"] == pytest.approx(interval["high"])

        varied = [
            record(s, 0.0, {ANALYTIC: matrix(c01=0.03) if s % 2 else matrix()}) for s in range(10)
        ]
        wide = only(significance_report(varied, seed=0), test=ANALYTIC)["null_rate"][0.05]
        assert wide["low"] < wide["estimate"] < wide["high"]

    def test_the_interval_is_over_datasets_not_cells(self):
        # 10 datasets x 6 cells: a cell-wise Wilson interval for 5 / 60 would be about
        # [0.04, 0.18]; clusters of cells must give something at least as wide
        varied = [
            record(s, 0.0, {ANALYTIC: matrix(c01=0.03, c02=0.03) if s < 2 else matrix()})
            for s in range(10)
        ]
        interval = only(significance_report(varied, seed=0), test=ANALYTIC)["null_rate"][0.05]
        assert interval["estimate"] == pytest.approx(4 / 60)
        assert interval["low"] == pytest.approx(0.0, abs=1e-9)  # a bootstrap sample with neither

    def test_it_is_reproducible_and_depends_on_the_seed(self):
        # 40 datasets with 0 to 4 rejected cells each: enough different values for the
        # percentiles of two bootstraps to differ
        varied = [
            record(
                s,
                0.0,
                {
                    ANALYTIC: matrix(
                        **{f"c{i}{j}": 0.03 for i, j in [(0, 1), (0, 2), (1, 0), (1, 2)][: s % 5]}
                    )
                },
            )
            for s in range(40)
        ]
        a = significance_report(varied, seed=1, n_boot=200)
        b = significance_report(varied, seed=1, n_boot=200)
        c = significance_report(varied, seed=2, n_boot=200)
        assert a == b
        assert a[0]["null_rate"][0.05] != c[0]["null_rate"][0.05]
        assert a[0]["null_rate"][0.05]["estimate"] == c[0]["null_rate"][0.05]["estimate"]

    def test_a_dataset_level_proportion_has_a_wilson_interval(self):
        records = [
            record(s, 0.0, {ANALYTIC: matrix(c01=0.001) if s < 2 else matrix()}) for s in range(10)
        ]
        any_false = only(significance_report(records), test=ANALYTIC)["bh"]["any_false_rejection"]
        # Wilson 95% interval for 2 of 10
        assert any_false["estimate"] == pytest.approx(0.2)
        assert any_false["low"] == pytest.approx(0.0567, abs=2e-3)
        assert any_false["high"] == pytest.approx(0.5098, abs=2e-3)


class TestGrouping:
    def test_snr_coupling_test_and_estimator_make_separate_rows(self):
        records = []
        for snr in (10.0, 40.0):
            for coupling in (0.0, 0.4):
                records.append(
                    record(0, coupling, {ANALYTIC: matrix(), SURROGATE: matrix()}, snr=snr)
                )
        records.append(record(0, 0.0, {ANALYTIC: matrix()}, estimator="gc_pairwise/fixed"))
        rows = significance_report(records)
        assert len(rows) == 4 * 2 + 1

    def test_failed_records_other_scenarios_and_records_without_tests_are_ignored(self):
        good = record(0, 0.0, {ANALYTIC: matrix(c01=0.001)})
        failed = record(1, 0.0, {ANALYTIC: matrix(c01=0.001)}, status="failed")
        other = record(2, 0.0, {ANALYTIC: matrix(c01=0.001)}, scenario="pair:a-lh:b-lh")
        scores_only = record(3, 0.0, {})
        scores_only["result"]["gc/fixed"].pop("tests")
        rows = significance_report([good, failed, other, scores_only])
        assert only(rows, test=ANALYTIC)["n_datasets"] == 1

    def test_the_diagonal_is_not_a_cell(self):
        row = only(significance_report([record(0, 0.0, {ANALYTIC: matrix()})]), test=ANALYTIC)
        assert row["n_null_cells"] == 6

    def test_no_records_gives_no_rows(self):
        assert significance_report([]) == []

    def test_nodes_that_are_not_those_of_the_scenario_are_an_error_not_a_skipped_record(self):
        bad = record(0, 0.0, {ANALYTIC: matrix()})
        bad["result"]["gc/fixed"]["analysed"] = ["a", "b", "c"]
        with pytest.raises(ValueError, match="not nodes of the scenario"):
            significance_report([bad])


class TestRendering:
    def header(self, n_surrogates=39):
        return {
            "protocol": "significance",
            "protocol_kind": "significance",
            "environment": ENV,
            "settings": {
                "estimators": ["gc/fixed"],
                "settings": {"n_surrogates": n_surrogates},
                "tests": [ANALYTIC, SURROGATE],
            },
        }

    def records(self):
        both = lambda m: {ANALYTIC: m, SURROGATE: m}  # noqa: E731
        return [
            record(s, 0.0, both(matrix(c01=0.03) if s == 0 else matrix())) for s in range(4)
        ] + [record(s, 0.4, both(matrix(c01=0.0001, c12=0.0001))) for s in range(4)]

    def test_the_summary_has_a_section_with_one_table_row_per_group(self):
        text = render_summary(self.header(), self.records(), None)
        assert "## Significance study" in text
        assert "| analytic |" in text and "| surrogate:phase_randomize |" in text

    def test_a_level_that_cannot_be_reached_says_so_instead_of_zero(self):
        def table(n_surrogates):
            text = render_summary(self.header(n_surrogates), self.records(), None)
            section = text.split("## Significance study")[1]
            return [line for line in section.splitlines() if line.startswith("|")]

        few = table(39)
        assert any("n/a" in line and "surrogate:phase_randomize" in line for line in few)
        assert not any("n/a" in line and "| analytic |" in line for line in few)
        # with 199 surrogates the smallest p-value is 0.005, so every level is reachable
        assert not any("n/a" in line for line in table(199))

    def test_the_edge_metric_tables_of_the_benchmark_are_not_shown(self):
        text = render_summary(self.header(), self.records(), None)
        assert "| mean AUROC |" not in text and "ROI vs sensor" not in text

    def test_it_reports_the_expected_mean_p_value_of_each_test(self):
        text = render_summary(self.header(39), self.records(), None)
        assert "0.512" in text and "0.500" in text  # 41 / 80 for the surrogate test, 0.5 analytic


# --- frequency-resolved tests (PDC, rPDC) -------------------------------------

FREQS = [float(f) for f in range(51)]  # 0 to 50 Hz in steps of 1 Hz: alpha 6 points, beta 18


def spectrum(fill=0.5, **cells):
    """A 3 x 3 x 51 p-value array, NaN diagonal; `cells` sets c01={10: 0.01} (frequency: p)."""
    out = np.full((3, 3, len(FREQS)), fill)
    for i in range(3):
        out[i, i] = NAN
    for key, values in cells.items():
        i, j = int(key[1]), int(key[2])
        if isinstance(values, dict):
            for freq, value in values.items():
                out[i, j, FREQS.index(float(freq))] = value
        else:
            out[i, j] = values
    return out.tolist()


def frecord(seed, coupling, tests, *, estimator="pdc/fixed", **kwargs):
    """A task record of a spectral estimator: per-frequency p-values and the frequency grid."""
    out = record(seed, coupling, tests, estimator=estimator, **kwargs)
    out["result"][estimator]["freqs"] = FREQS
    return out


class TestFrequencyNullWorld:
    def records(self):
        return [
            frecord(0, 0.0, {ANALYTIC: spectrum(c01={10: 0.01, 20: 0.03})}),
            frecord(1, 0.0, {ANALYTIC: spectrum(c21={20: 0.001})}),
        ]

    def test_there_is_a_row_for_the_whole_grid_and_one_for_each_band(self):
        rows = frequency_report(self.records())
        assert [(r["band"], r["fmin"], r["fmax"], r["n_freqs"]) for r in rows] == [
            ("all", 0.0, 50.0, 51),
            ("alpha", 8.0, 13.0, 6),
            ("beta", 13.0, 30.0, 18),
        ]
        assert all(r["n_null_pairs"] == 12 and r["n_edge_pairs"] == 0 for r in rows)

    def test_the_pointwise_rate_is_over_null_pairs_and_the_frequencies_of_the_band(self):
        rows = frequency_report(self.records())
        whole = only(rows, band="all")
        assert whole["null_rate"][0.05]["estimate"] == pytest.approx(3 / (12 * 51))
        assert whole["null_rate"][0.01]["estimate"] == pytest.approx(2 / (12 * 51))
        assert only(rows, band="alpha")["null_rate"][0.05]["estimate"] == pytest.approx(1 / 72)
        assert only(rows, band="beta")["null_rate"][0.05]["estimate"] == pytest.approx(2 / 216)
        assert whole["power"][0.05] is None

    def test_after_bh_within_a_band_a_pair_is_a_false_positive_if_any_frequency_survives(self):
        rows = frequency_report(self.records())
        # beta: 0.001 * 18 = 0.018 survives; 0.03 * 18 does not
        beta = only(rows, band="beta")["pair_bh"]["null_rate"]
        assert beta["estimate"] == pytest.approx(1 / 12)
        # alpha: 0.01 * 6 = 0.06 does not survive; the whole grid: 0.001 * 51 = 0.051 does not
        assert only(rows, band="alpha")["pair_bh"]["null_rate"]["estimate"] == 0.0
        assert only(rows, band="all")["pair_bh"]["null_rate"]["estimate"] == 0.0

    def test_both_limits_of_a_band_are_inclusive(self):
        rows = frequency_report([frecord(0, 0.0, {ANALYTIC: spectrum(c02={13: 0.02})})])
        assert only(rows, band="alpha")["null_rate"][0.05]["estimate"] > 0
        assert only(rows, band="beta")["null_rate"][0.05]["estimate"] > 0

    def test_the_profile_is_the_pointwise_rate_at_each_frequency(self):
        row = only(frequency_report(self.records()), band="all")
        assert row["null_profile"]["freqs"] == FREQS
        assert row["null_profile"]["rate"][10] == pytest.approx(1 / 12)
        assert row["null_profile"]["rate"][20] == pytest.approx(2 / 12)
        assert row["null_profile"]["rate"][0] == 0.0

    def test_other_bands_can_be_asked_for_and_an_unknown_one_is_named(self):
        rows = frequency_report(self.records(), bands=("theta",))
        assert [r["band"] for r in rows] == ["all", "theta"]
        with pytest.raises(ValueError, match="betta"):
            frequency_report(self.records(), bands=("betta",))


class TestFrequencyLiberal:
    def test_a_test_is_liberal_when_the_lower_end_of_its_interval_is_above_the_level(self):
        bad = [frecord(s, 0.0, {ANALYTIC: spectrum(fill=0.01)}) for s in range(5)]
        good = [frecord(s, 0.0, {ANALYTIC: spectrum()}) for s in range(5)]
        assert only(frequency_report(bad), band="all")["liberal"][0.05] is True
        assert only(frequency_report(good), band="all")["liberal"][0.05] is False

    def test_a_level_that_cannot_be_reached_has_no_verdict(self):
        records = [frecord(0, 0.0, {SURROGATE: spectrum()})]
        row = only(frequency_report(records, n_surrogates=39), band="all")
        assert row["null_rate"][0.01] is None and row["liberal"][0.01] is None


class TestFrequencyEdges:
    def test_power_pointwise_and_per_pair_after_bh(self):
        records = [
            frecord(0, 0.4, {ANALYTIC: spectrum(c01=0.0001, c12=0.0001)}),
            frecord(1, 0.4, {ANALYTIC: spectrum(c01=0.0001)}),
        ]
        row = only(frequency_report(records), band="beta")
        assert row["n_edge_pairs"] == 4 and row["n_null_pairs"] == 8
        assert row["power"][0.05]["estimate"] == pytest.approx(3 / 4)
        assert row["pair_bh"]["power"]["estimate"] == pytest.approx(3 / 4)

    def test_bh_within_a_band_needs_several_frequencies_at_the_smallest_surrogate_p_value(self):
        records = [frecord(0, 0.4, {ANALYTIC: spectrum(), SURROGATE: spectrum()})]
        rows = frequency_report(records, n_surrogates=39)
        # p_min = 1 / 40: 0.025 * 18 / k <= 0.05 needs k >= 9 in beta, 3 in alpha
        assert only(rows, band="beta", test=SURROGATE)["bh_freqs_needed"] == 9
        assert only(rows, band="alpha", test=SURROGATE)["bh_freqs_needed"] == 3
        assert only(rows, band="beta", test=ANALYTIC)["bh_freqs_needed"] == 1


class TestTwoKindsOfRecord:
    def test_each_report_takes_only_its_own_kind(self):
        cells = record(0, 0.0, {ANALYTIC: matrix()})
        spectra = frecord(0, 0.0, {ANALYTIC: spectrum()})
        assert len(significance_report([cells, spectra])) == 1
        assert {r["estimator"] for r in frequency_report([cells, spectra])} == {"pdc/fixed"}

    def test_estimators_tests_and_conditions_make_separate_groups(self):
        records = [
            frecord(0, c, {ANALYTIC: spectrum(), SURROGATE: spectrum()}, estimator=e, snr=snr)
            for c in (0.0, 0.4)
            for snr in (10.0, 40.0)
            for e in ("pdc/fixed", "rpdc/low")
        ]
        assert len(frequency_report(records)) == 2 * 2 * 2 * 2 * 3

    def test_the_summary_has_a_frequency_resolved_section(self):
        header = TestRendering().header()
        records = [frecord(s, 0.0, {ANALYTIC: spectrum(), SURROGATE: spectrum()}) for s in range(3)]
        text = render_summary(header, records, None)
        section = text.split("Frequency-resolved tests")[1]
        assert "| pdc/fixed | analytic | 3 | all |" in section
        assert "| beta |" in section and "| alpha |" in section
