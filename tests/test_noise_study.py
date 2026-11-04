"""Analysis of the noise study: the three arms side by side, paired by seed."""

import math

import pytest

from dirconn.noise_study import noise_report
from dirconn.protocol import render_summary

ENV = {"python": "3.11", "numpy": "2.4.6"}


def record(noise, seed, auroc, direction=1.0, *, modality="grad", estimator="gc/bic", snr=-10.0):
    """A task record of the chain whose sensor stage scores `auroc` (the other stages 1.0)."""
    task = {
        "scenario": "chain",
        "modality": modality,
        "snr_db": snr,
        "duration_s": 30.0,
        "coupling": None,
        "source_model": "partial",
        "seed_index": seed,
    }
    if noise != "simulated":
        task["noise"] = noise  # the default is left out of a task's key
    perfect = {"auroc": 1.0, "direction_accuracy": 1.0}
    return {
        "type": "task",
        "status": "ok",
        "id": f"{noise}-{seed}-{modality}-{estimator}-{snr}",
        "task": task,
        "result": {
            estimator: {
                "roi": perfect,
                "sensor": {"auroc": auroc, "direction_accuracy": direction},
                "source": perfect,
            }
        },
    }


def three_arms(**kw):
    sim, white, empty = [1.0, 0.8, 0.6], [0.9, 0.7, 0.5], [0.5, 0.5, 0.5]
    return (
        [record("simulated", s, a, **kw) for s, a in enumerate(sim)]
        + [record("white", s, a, **kw) for s, a in enumerate(white)]
        + [record("empty_room", s, a, **kw) for s, a in enumerate(empty)]
    )


def only(rows, **where):
    (row,) = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return row


class TestArms:
    def test_the_mean_of_each_arm_and_the_number_of_seeds(self):
        row = only(noise_report(three_arms()), stage="sensor")
        assert row["arms"]["simulated"]["auroc"] == pytest.approx(0.8)
        assert row["arms"]["white"]["auroc"] == pytest.approx(0.7)
        assert row["arms"]["empty_room"]["auroc"] == pytest.approx(0.5)
        assert {k: v["n"] for k, v in row["arms"].items()} == {
            "simulated": 3,
            "white": 3,
            "empty_room": 3,
        }

    def test_the_direction_accuracy_is_averaged_too(self):
        records = [record("simulated", s, 1.0, direction=d) for s, d in enumerate([1.0, 0.5])]
        row = only(noise_report(records), stage="sensor")
        assert row["arms"]["simulated"]["direction_accuracy"] == pytest.approx(0.75)

    def test_the_arms_come_in_a_fixed_order(self):
        row = only(noise_report(list(reversed(three_arms()))), stage="sensor")
        assert list(row["arms"]) == ["simulated", "white", "empty_room"]

    def test_a_task_without_a_noise_key_is_the_simulated_noise(self):
        rows = noise_report([record("simulated", 0, 0.9)])
        assert list(only(rows, stage="sensor")["arms"]) == ["simulated"]


class TestPairedDifferences:
    def test_the_three_differences_are_taken_seed_by_seed(self):
        paired = only(noise_report(three_arms()), stage="sensor")["paired"]
        # white - simulated = -0.1 for every seed; empty_room - white = -0.4, -0.2, 0.0
        assert paired["white_minus_simulated"]["auroc"]["mean"] == pytest.approx(-0.1)
        assert paired["white_minus_simulated"]["auroc"]["sd"] == pytest.approx(0.0, abs=1e-12)
        assert paired["empty_room_minus_white"]["auroc"]["mean"] == pytest.approx(-0.2)
        assert paired["empty_room_minus_white"]["auroc"]["sd"] == pytest.approx(0.2)  # ddof 1
        assert paired["empty_room_minus_simulated"]["auroc"]["mean"] == pytest.approx(-0.3)
        assert paired["empty_room_minus_simulated"]["n"] == 3

    def test_only_seeds_that_both_arms_have_are_paired(self):
        records = three_arms()
        records = [
            r
            for r in records
            if not (r["task"].get("noise") == "empty_room" and r["task"]["seed_index"] == 2)
        ]
        row = only(noise_report(records), stage="sensor")
        assert row["arms"]["empty_room"]["n"] == 2
        assert row["paired"]["empty_room_minus_white"]["n"] == 2
        assert row["paired"]["empty_room_minus_white"]["auroc"]["mean"] == pytest.approx(-0.3)
        assert row["paired"]["white_minus_simulated"]["n"] == 3

    def test_a_pair_with_an_undefined_auroc_is_left_out(self):
        records = three_arms()
        records[0]["result"]["gc/bic"]["sensor"]["auroc"] = math.nan  # simulated, seed 0
        row = only(noise_report(records), stage="sensor")
        assert row["paired"]["white_minus_simulated"]["n"] == 2
        assert row["paired"]["white_minus_simulated"]["auroc"]["mean"] == pytest.approx(-0.1)
        assert row["arms"]["simulated"]["n"] == 2
        assert row["arms"]["simulated"]["auroc"] == pytest.approx(0.7)

    def test_a_single_pair_has_no_standard_deviation(self):
        records = [record("simulated", 0, 1.0), record("white", 0, 0.5)]
        entry = only(noise_report(records), stage="sensor")["paired"]["white_minus_simulated"]
        assert entry["n"] == 1 and entry["auroc"]["mean"] == pytest.approx(-0.5)
        assert math.isnan(entry["auroc"]["sd"])

    def test_with_one_arm_there_is_nothing_to_pair(self):
        rows = noise_report([record("white", s, 0.5) for s in range(3)])
        assert only(rows, stage="sensor")["paired"] == {}

    def test_the_difference_is_in_the_direction_of_the_name(self):
        records = [record("simulated", 0, 0.2), record("white", 0, 0.9)]
        entry = only(noise_report(records), stage="sensor")["paired"]["white_minus_simulated"]
        assert entry["auroc"]["mean"] == pytest.approx(0.7)  # white minus simulated, positive


class TestGrouping:
    def test_modality_estimator_snr_and_stage_make_separate_rows(self):
        records = three_arms() + three_arms(modality="mag") + three_arms(estimator="gpdc/bic")
        records += three_arms(snr=0.0)
        rows = noise_report(records)
        assert len(rows) == 4 * 3  # four groups, three stages each
        assert {r["stage"] for r in rows} == {"roi", "sensor", "source"}

    def test_failed_records_and_other_kinds_of_task_are_ignored(self):
        records = three_arms()
        failed = record("white", 5, 0.0)
        failed["status"] = "failed"
        pair = record("white", 6, 0.0)
        pair["task"]["scenario"] = "pair:a-lh:b-lh"
        records += [failed, pair]
        assert only(noise_report(records), stage="sensor")["arms"]["white"]["n"] == 3

    def test_no_records_gives_no_rows(self):
        assert noise_report([]) == []


class TestRendering:
    def header(self):
        return {
            "protocol": "noise",
            "protocol_kind": "scenarios",
            "environment": ENV,
            "settings": {"estimators": ["gc/bic"]},
        }

    def test_the_summary_has_a_section_with_the_arms_and_the_differences(self):
        text = render_summary(self.header(), three_arms(), None)
        assert "## Noise study" in text
        section = text.split("## Noise study")[1]
        assert "simulated" in section and "white" in section and "empty room" in section
        assert "0.800" in section and "0.700" in section and "0.500" in section  # the arm means
        assert "-0.300" in section  # empty room minus simulated, as the mean difference

    def test_it_says_that_the_noise_windows_are_reused_and_how_long_the_recording_is(self):
        section = render_summary(self.header(), three_arms(), None).split("## Noise study")[1]
        assert "reused" in section and "110 s" in section
        assert "not independent" in section

    def test_a_run_with_only_the_simulated_noise_has_no_such_section(self):
        records = [record("simulated", s, 0.9) for s in range(3)]
        assert "## Noise study" not in render_summary(self.header(), records, None)
