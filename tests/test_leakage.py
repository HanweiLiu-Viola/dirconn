"""Analysis of the leakage studies: crosstalk vs false edges, and the neighbour."""

import itertools
import math

import numpy as np
import pytest

from dirconn.leakage import adjacent_report, leakage_correlation

NAMES = ["bankssts-lh", "cuneus-lh", "insula-lh", "lingual-lh"]
PAIRS = list(itertools.combinations(NAMES, 2))  # 6 unordered pairs


def crosstalk_matrix(seed=0):
    """matrix[y, x]: the leak of region x into the series of region y; asymmetric on purpose."""
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.05, 3.0, size=(4, 4))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def header(matrix=None, modalities=("eeg",)):
    matrix = crosstalk_matrix() if matrix is None else matrix
    return {
        "protocol": "leakage_pairs",
        "protocol_kind": "region_pairs",
        "crosstalk": {m: {"names": NAMES, "matrix": matrix.tolist()} for m in modalities},
    }


def card(scores):
    return {"auroc": float("nan"), "direction_accuracy": float("nan"), "scores": scores}


def pair_record(x, y, scores_by_stage, *, modality="eeg", estimator="gc/bic", status="ok"):
    """A region-pair task whose cards hold the given 2 x 2 score matrices."""
    return {
        "type": "task",
        "status": status,
        "id": f"{x}-{y}-{modality}",
        "task": {
            "scenario": f"pair:{x}:{y}",
            "modality": modality,
            "snr_db": 10.0,
            "duration_s": 60.0,
            "coupling": 0.0,
            "source_model": "partial",
            "seed_index": 0,
        },
        "result": {
            estimator: {
                "analysed": [x, y],
                **{stage: card(s) for stage, s in scores_by_stage.items()},
            }
        },
    }


def faithful_records(matrix, *, modality="eeg", estimator="gc/bic", noise=None):
    """Scores that rise exactly with the leak: x -> y scores |m[y, x]|, y -> x scores |m[x, y]|."""
    records = []
    for x, y in PAIRS:
        i, j = NAMES.index(x), NAMES.index(y)
        source = [[math.nan, abs(matrix[j, i])], [abs(matrix[i, j]), math.nan]]
        records.append(
            pair_record(
                x,
                y,
                {"source": source, "roi": [[math.nan, 0.5], [0.5, math.nan]]},
                modality=modality,
                estimator=estimator,
            )
        )
    return records


class TestLeakageCorrelation:
    def test_scores_that_follow_the_leak_give_a_correlation_of_one(self):
        matrix = crosstalk_matrix()
        rows = leakage_correlation(header(matrix), faithful_records(matrix), n_boot=200, seed=0)
        (row,) = [r for r in rows if r["stage"] == "source"]
        assert row["spearman"] == pytest.approx(1.0)
        assert row["n_cells"] == 12 and row["n_pairs"] == 6
        assert row["ci_low"] == pytest.approx(1.0) and row["ci_high"] == pytest.approx(1.0)

    def test_a_stage_without_the_leak_has_no_correlation(self):
        matrix = crosstalk_matrix()
        rows = leakage_correlation(header(matrix), faithful_records(matrix), n_boot=50, seed=0)
        (row,) = [r for r in rows if r["stage"] == "roi"]
        assert math.isnan(row["spearman"])  # every score is the same: no rank information

    def test_the_cell_x_to_y_is_paired_with_the_leak_of_x_into_y_not_the_other_way_round(self):
        matrix = crosstalk_matrix()
        # scores that follow the transposed matrix: they are the leak of y into x, so a method
        # that pairs the cells the wrong way round would get a perfect correlation from them
        records = faithful_records(matrix.T)
        rows = leakage_correlation(header(matrix), records, n_boot=50, seed=0)
        (row,) = [r for r in rows if r["stage"] == "source"]
        assert row["spearman"] < 0.9

    def test_the_sign_of_the_crosstalk_is_ignored(self):
        matrix = crosstalk_matrix()
        signs = np.where(np.random.default_rng(5).random((4, 4)) < 0.5, -1.0, 1.0)
        rows = leakage_correlation(
            header(matrix * signs), faithful_records(matrix), n_boot=50, seed=0
        )
        assert [r for r in rows if r["stage"] == "source"][0]["spearman"] == pytest.approx(1.0)

    def test_one_row_per_estimator_modality_and_stage(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix) + faithful_records(matrix, estimator="dtf/bic")
        records += faithful_records(matrix, modality="grad")
        rows = leakage_correlation(header(matrix, ("eeg", "grad")), records, n_boot=20, seed=0)
        keys = {(r["estimator"], r["modality"], r["stage"]) for r in rows}
        assert keys == {
            (e, m, s)
            for e, m in (("gc/bic", "eeg"), ("dtf/bic", "eeg"), ("gc/bic", "grad"))
            for s in ("roi", "source")
        }

    def test_the_interval_is_ordered_contains_the_estimate_and_depends_on_the_seed(self):
        matrix = crosstalk_matrix()
        rng = np.random.default_rng(2)
        records = []
        for x, y in PAIRS:
            i, j = NAMES.index(x), NAMES.index(y)
            noisy = [
                [math.nan, abs(matrix[j, i]) + rng.normal(0, 1.0)],
                [abs(matrix[i, j]) + rng.normal(0, 1.0), math.nan],
            ]
            records.append(pair_record(x, y, {"source": noisy}))
        a = leakage_correlation(header(matrix), records, n_boot=400, seed=1)[0]
        b = leakage_correlation(header(matrix), records, n_boot=400, seed=1)[0]
        c = leakage_correlation(header(matrix), records, n_boot=400, seed=2)[0]
        assert a == b
        assert a["ci_low"] <= a["spearman"] <= a["ci_high"]
        assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
        assert a["ci_high"] - a["ci_low"] > 0.1  # noisy scores: a real interval, not a point

    def test_nan_scores_are_left_out_and_counted_out(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix)
        records[0]["result"]["gc/bic"]["source"]["scores"][0][1] = math.nan
        (row,) = [
            r
            for r in leakage_correlation(header(matrix), records, n_boot=20, seed=0)
            if r["stage"] == "source"
        ]
        assert row["n_cells"] == 11 and row["n_pairs"] == 6

    def test_failed_records_and_other_scenarios_are_ignored(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix)
        records.append(
            pair_record(*PAIRS[0], {"source": [[math.nan, 99.0], [0.0, math.nan]]}, status="failed")
        )
        other = pair_record(*PAIRS[1], {"source": [[math.nan, 99.0], [0.0, math.nan]]})
        other["task"]["scenario"] = "chain"
        records.append(other)
        (row,) = [
            r
            for r in leakage_correlation(header(matrix), records, n_boot=20, seed=0)
            if r["stage"] == "source"
        ]
        assert row["n_cells"] == 12 and row["spearman"] == pytest.approx(1.0)

    def test_a_modality_without_a_crosstalk_matrix_is_named(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix, modality="mag")
        with pytest.raises(ValueError, match="mag"):
            leakage_correlation(header(matrix), records, n_boot=10, seed=0)

    def test_a_region_missing_from_the_matrix_is_named(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix)
        records[0]["task"]["scenario"] = "pair:nowhere-lh:insula-lh"
        records[0]["result"]["gc/bic"]["analysed"] = ["nowhere-lh", "insula-lh"]
        with pytest.raises(ValueError, match="nowhere-lh"):
            leakage_correlation(header(matrix), records, n_boot=10, seed=0)

    def test_several_seeds_of_a_pair_count_as_one_pair_in_the_resampling(self):
        matrix = crosstalk_matrix()
        records = faithful_records(matrix)
        again = faithful_records(matrix)
        for record in again:
            record["task"]["seed_index"] = 1
            record["id"] += "-b"
        (row,) = [
            r
            for r in leakage_correlation(header(matrix), records + again, n_boot=20, seed=0)
            if r["stage"] == "source"
        ]
        assert row["n_cells"] == 24 and row["n_pairs"] == 6

    def test_no_records_gives_no_rows(self):
        assert leakage_correlation(header(), [], n_boot=10, seed=0) == []


# ---- the neighbour scenario ----------------------------------------------------------

Z, X, Y = "transversetemporal-rh", "cuneus-lh", "pericalcarine-lh"


def adjacent_record(seed, *, roi, source, snr=10.0, modality="eeg", estimator="gc/bic", auroc=None):
    """A leakage_adjacent task. `roi` and `source` are (z->x, z->y, x->y) scores."""

    def matrix(z_x, z_y, x_y):
        nan = math.nan
        return [[nan, z_x, z_y], [0.01, nan, x_y], [0.01, 0.01, nan]]

    a = {"roi": 1.0, "source": 0.8} if auroc is None else auroc
    return {
        "type": "task",
        "status": "ok",
        "id": f"adj-{seed}-{snr}-{modality}-{estimator}",
        "task": {
            "scenario": "leakage_adjacent",
            "modality": modality,
            "snr_db": snr,
            "duration_s": 60.0,
            "coupling": None,
            "source_model": "partial",
            "seed_index": seed,
        },
        "result": {
            estimator: {
                "analysed": [Z, X, Y],
                "roi": {
                    "auroc": a["roi"],
                    "direction_accuracy": 1.0,
                    "scores": matrix(*roi),
                },
                "source": {
                    "auroc": a["source"],
                    "direction_accuracy": 0.75,
                    "scores": matrix(*source),
                },
            }
        },
    }


class TestAdjacentReport:
    def records(self):
        # relative false-edge scores: the false cell divided by the true edge Z -> X
        return [
            adjacent_record(0, roi=(1.0, 0.0, 0.0), source=(1.0, 0.2, 0.6)),
            adjacent_record(1, roi=(2.0, 0.2, 0.0), source=(2.0, 0.8, 1.0)),
        ]

    def test_it_reports_the_false_edge_scores_relative_to_the_true_edge(self):
        rows = adjacent_report(self.records())
        by_stage = {r["stage"]: r for r in rows}
        source = by_stage["source"]
        # seed 0: Z->Y 0.2, X->Y 0.6; seed 1: Z->Y 0.4, X->Y 0.5
        assert source["z_to_y"]["mean"] == pytest.approx(0.3)
        assert source["z_to_y"]["min"] == pytest.approx(0.2)
        assert source["z_to_y"]["max"] == pytest.approx(0.4)
        assert source["x_to_y"]["mean"] == pytest.approx(0.55)
        roi = by_stage["roi"]
        assert roi["z_to_y"]["mean"] == pytest.approx(0.05)
        assert roi["x_to_y"]["mean"] == pytest.approx(0.0)

    def test_it_reports_auroc_and_direction_accuracy(self):
        rows = {r["stage"]: r for r in adjacent_report(self.records())}
        assert rows["roi"]["auroc"] == pytest.approx(1.0)
        assert rows["source"]["auroc"] == pytest.approx(0.8)
        assert rows["source"]["direction_accuracy"] == pytest.approx(0.75)
        assert rows["source"]["n"] == 2

    def test_the_source_minus_roi_difference_is_paired_by_seed(self):
        records = [
            adjacent_record(
                0, roi=(1.0, 0.0, 0.0), source=(1.0, 0.2, 0.2), auroc={"roi": 1.0, "source": 0.9}
            ),
            adjacent_record(
                1, roi=(1.0, 0.0, 0.0), source=(1.0, 0.6, 0.6), auroc={"roi": 1.0, "source": 0.5}
            ),
        ]
        (diff,) = adjacent_report(records, paired=True)
        assert diff["auroc"]["mean"] == pytest.approx(-0.3)
        assert diff["auroc"]["sd"] == pytest.approx(np.std([-0.1, -0.5], ddof=1))
        assert diff["z_to_y"]["mean"] == pytest.approx(0.4)
        assert diff["n"] == 2

    def test_groups_are_kept_apart_by_estimator_modality_and_snr(self):
        records = self.records()
        records += [adjacent_record(0, roi=(1.0, 0.0, 0.0), source=(1.0, 0.9, 0.9), snr=0.0)]
        records += [
            adjacent_record(0, roi=(1.0, 0.0, 0.0), source=(1.0, 0.1, 0.1), modality="grad")
        ]
        records += [
            adjacent_record(0, roi=(1.0, 0.0, 0.0), source=(1.0, 0.5, 0.5), estimator="dtf/bic")
        ]
        keys = {
            (r["estimator"], r["modality"], r["snr_db"], r["stage"])
            for r in adjacent_report(records)
        }
        assert len(keys) == 4 * 2  # four groups, two stages each

    def test_a_true_edge_score_of_zero_or_nan_gives_nan_relative_scores(self):
        records = [adjacent_record(0, roi=(0.0, 0.1, 0.1), source=(math.nan, 0.1, 0.1))]
        rows = {r["stage"]: r for r in adjacent_report(records)}
        assert math.isnan(rows["roi"]["z_to_y"]["mean"])
        assert math.isnan(rows["source"]["z_to_y"]["mean"])

    def test_failed_records_and_other_scenarios_are_ignored(self):
        records = self.records()
        records.append({**adjacent_record(2, roi=(1, 9, 9), source=(1, 9, 9)), "status": "failed"})
        other = adjacent_record(3, roi=(1, 9, 9), source=(1, 9, 9))
        other["task"]["scenario"] = "chain"
        records.append(other)
        rows = {r["stage"]: r for r in adjacent_report(records)}
        assert rows["source"]["n"] == 2

    def test_no_records_gives_no_rows(self):
        assert adjacent_report([]) == []
