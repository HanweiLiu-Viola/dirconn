"""The generated benchmark report: flat rows, grids, paired differences, run time."""

import math
from pathlib import Path

import pytest

from dirconn.report import (
    ADDONS,
    PUBLISHED,
    build_report,
    flatten,
    grid_markdown,
    load_published,
    mean_grid,
    paired_differences,
    runtime_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
ENV = {"python": "3.11.16", "platform": "Windows", "numpy": "2.4.6", "scipy": "1.17.1"}


def card(auroc, direction=1.0, separation=2.0):
    return {"auroc": auroc, "direction_accuracy": direction, "separation": separation}


def record(
    seed,
    snr,
    values,
    *,
    scenario="chain",
    duration=30.0,
    coupling=None,
    source_model="partial",
    modality="eeg",
    status="ok",
):
    """A task record; `values` maps a label (``gc/bic@lcmv`` too) to (roi, sensor, source)."""
    return {
        "type": "task",
        "status": status,
        "id": f"{scenario}-{snr}-{duration}-{coupling}-{source_model}-{modality}-{seed}",
        "task": {
            "scenario": scenario,
            "snr_db": snr,
            "duration_s": duration,
            "coupling": coupling,
            "source_model": source_model,
            "modality": modality,
            "seed_index": seed,
        },
        "seconds": 1.0,
        "result": {
            label: {
                "roi": card(v[0]),
                "sensor": card(v[1]),
                "source": card(v[2]),
                "agreement": {"spearman": 0.5},
            }
            for label, v in values.items()
        },
    }


def header(name="full", **extra):
    return {
        "protocol": name,
        "environment": ENV,
        "gate": {"estimate_hours": 0.4, "workers": 6},
        "settings": {"estimators": ["gc/bic", "gc/fixed"]},
        **extra,
    }


class TestFlatten:
    def test_one_row_per_task_estimator_and_stage_with_the_fields_split_out(self):
        rows = flatten(
            "full", [record(0, 0.0, {"gc/bic": (1.0, 0.9, 0.8), "dtf/fixed": (1, 1, 1)})]
        )
        assert len(rows) == 2 * 3
        first = [r for r in rows if r["label"] == "gc/bic" and r["stage"] == "sensor"][0]
        assert first["method"] == "gc" and first["order"] == "bic" and first["inverse"] is None
        assert first["auroc"] == 0.9 and first["protocol"] == "full" and first["snr_db"] == 0.0
        assert first["seed_index"] == 0 and first["scenario"] == "chain"

    def test_the_inverse_method_is_split_from_the_label(self):
        rows = flatten("inverse", [record(0, 0.0, {"gc/bic@lcmv": (1, 1, 0.5)})])
        assert {r["inverse"] for r in rows} == {"lcmv"} and {r["label"] for r in rows} == {"gc/bic"}

    def test_the_coupling_of_a_task_is_its_strength_the_file_strength_when_unset(self):
        rows = flatten(
            "x",
            [
                record(0, 0.0, {"gc/bic": (1, 1, 1)}, coupling=0.1),
                record(1, 0.0, {"gc/bic": (1, 1, 1)}, coupling=None),
                record(2, 0.0, {"gc/bic": (1, 1, 1)}, scenario="random_5"),
            ],
        )
        by_seed = {r["seed_index"]: r["coupling"] for r in rows if r["stage"] == "roi"}
        assert by_seed[0] == 0.1
        assert by_seed[1] == 0.4  # the strength of every edge of the chain
        assert by_seed[2] is None  # a graph with edges of different strengths has no single one

    def test_failed_tasks_are_left_out_and_a_missing_noise_is_the_simulated_one(self):
        rows = flatten(
            "x",
            [
                record(0, 0.0, {"gc/bic": (1, 1, 1)}),
                record(1, 0.0, {"gc/bic": (0, 0, 0)}, status="failed"),
            ],
        )
        assert {r["seed_index"] for r in rows} == {0}
        assert {r["noise"] for r in rows} == {"simulated"}

    def test_the_task_id_is_kept_for_pairing(self):
        rows = flatten("x", [record(0, 0.0, {"gc/bic": (1, 1, 1)})])
        assert len({r["task_id"] for r in rows}) == 1


class TestMeanGrid:
    def rows(self):
        recs = []
        for seed, (a, b) in enumerate([(1.0, 0.5), (0.8, 0.5), (0.6, 0.5)]):
            recs.append(record(seed, 0.0, {"gc/bic": (1, a, 1), "dtf/bic": (1, b, 1)}))
        recs.append(record(0, -10.0, {"gc/bic": (1, 0.2, 1), "dtf/bic": (1, 0.1, 1)}))
        return flatten("full", recs)

    def test_cells_are_means_over_seeds_with_their_count(self):
        grid = mean_grid(
            self.rows(), stage="sensor", row_keys=("snr_db",), columns=("gc/bic", "dtf/bic")
        )
        assert grid[(0.0,)]["gc/bic"] == (pytest.approx(0.8), 3)
        assert grid[(0.0,)]["dtf/bic"] == (pytest.approx(0.5), 3)
        assert grid[(-10.0,)]["gc/bic"] == (pytest.approx(0.2), 1)

    def test_nan_values_are_left_out_of_the_mean_and_the_count(self):
        rows = self.rows()
        for r in rows:
            if r["seed_index"] == 1 and r["snr_db"] == 0.0 and r["label"] == "gc/bic":
                r["auroc"] = math.nan
        grid = mean_grid(rows, stage="sensor", row_keys=("snr_db",), columns=("gc/bic",))
        assert grid[(0.0,)]["gc/bic"] == (pytest.approx(0.8), 2)  # (1.0 + 0.6) / 2

    def test_a_cell_without_values_is_nan_with_zero_seeds(self):
        grid = mean_grid(
            self.rows(), stage="sensor", row_keys=("snr_db",), columns=("gc/bic", "pdc/bic")
        )
        mean, n = grid[(0.0,)]["pdc/bic"]
        assert math.isnan(mean) and n == 0

    def test_the_stage_and_the_row_keys_decide_the_rows(self):
        grid = mean_grid(
            self.rows(), stage="roi", row_keys=("scenario", "snr_db"), columns=("gc/bic",)
        )
        assert set(grid) == {("chain", 0.0), ("chain", -10.0)}
        assert grid[("chain", 0.0)]["gc/bic"][0] == pytest.approx(1.0)

    def test_the_markdown_has_one_row_per_key_and_a_cell_per_column(self):
        grid = mean_grid(
            self.rows(), stage="sensor", row_keys=("snr_db",), columns=("gc/bic", "dtf/bic")
        )
        text = grid_markdown(grid, row_names=("SNR",), columns=("gc/bic", "dtf/bic"))
        assert "| SNR | gc/bic | dtf/bic |" in text
        assert "| -10 | 0.200 | 0.100 |" in text and "| 0 | 0.800 | 0.500 |" in text

    def test_a_column_without_values_shows_a_dash_not_nan(self):
        columns = ("gc/bic", "pdc/bic")  # pdc/bic was not run in these rows
        grid = mean_grid(self.rows(), stage="sensor", row_keys=("snr_db",), columns=columns)
        text = grid_markdown(grid, row_names=("SNR",), columns=columns)
        assert "| 0 | 0.800 | - |" in text and "| -10 | 0.200 | - |" in text
        assert "nan" not in text


class TestPairedDifferences:
    def rows(self):
        recs = []
        for seed in range(4):
            # fixed beats bic by 0.1 for gc and by seed/10 for dtf
            recs.append(
                record(
                    seed,
                    0.0,
                    {
                        "gc/bic": (0.5, 0.5, 0.5),
                        "gc/fixed": (0.6, 0.6, 0.6),
                        "dtf/bic": (0.5, 0.5, 0.5),
                        "dtf/fixed": (0.5 + seed / 10, 0.5, 0.5),
                    },
                )
            )
        return flatten("full", recs)

    def test_the_order_setting_difference_is_taken_task_by_task(self):
        out = paired_differences(
            self.rows(),
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("method", "stage"),
        )
        assert out[("gc", "roi")]["mean"] == pytest.approx(0.1)
        assert out[("gc", "roi")]["sd"] == pytest.approx(0.0, abs=1e-12)
        assert out[("dtf", "roi")]["mean"] == pytest.approx(0.15)  # seeds 0..3: 0, .1, .2, .3
        assert out[("dtf", "roi")]["sd"] == pytest.approx(0.129099, abs=1e-5)  # ddof 1
        assert out[("dtf", "sensor")]["mean"] == pytest.approx(0.0)
        assert out[("gc", "roi")]["n"] == 4

    def test_only_pairs_that_exist_in_both_arms_count(self):
        rows = [r for r in self.rows() if not (r["label"] == "gc/fixed" and r["seed_index"] == 3)]
        out = paired_differences(
            rows,
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("method",),
        )
        assert out[("gc",)]["n"] == 3 * 3  # three seeds x three stages

    def test_a_difference_with_an_undefined_value_is_dropped(self):
        rows = self.rows()
        for r in rows:
            if r["label"] == "gc/bic" and r["seed_index"] == 0:
                r["auroc"] = math.nan
        out = paired_differences(
            rows,
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("method",),
        )
        assert out[("gc",)]["n"] == 3 * 3

    def test_the_sign_is_plus_minus_minus(self):
        out = paired_differences(
            self.rows(),
            field="order",
            plus="bic",
            minus="fixed",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("method",),
        )
        assert out[("gc",)]["mean"] == pytest.approx(-0.1)

    def test_stages_can_be_compared_for_the_same_estimator(self):
        recs = [record(s, 0.0, {"gc/bic": (1.0, 0.7, 0.9)}) for s in range(3)]
        out = paired_differences(
            flatten("full", recs),
            field="stage",
            plus="sensor",
            minus="roi",
            match=("task_id", "label"),
            group_by=("label",),
        )
        assert out[("gc/bic",)]["mean"] == pytest.approx(-0.3)

    def test_methods_can_be_compared_for_the_same_order(self):
        recs = [
            record(s, 0.0, {"gc/bic": (0.9, 0.9, 0.9), "gc_pairwise/bic": (0.6, 0.6, 0.6)})
            for s in range(3)
        ]
        out = paired_differences(
            flatten("full", recs),
            field="method",
            plus="gc",
            minus="gc_pairwise",
            match=("task_id", "order", "inverse", "stage"),
            group_by=("stage",),
        )
        assert out[("roi",)]["mean"] == pytest.approx(0.3)

    def test_a_single_pair_has_no_standard_deviation(self):
        recs = [record(0, 0.0, {"gc/bic": (0.5, 0.5, 0.5), "gc/fixed": (0.7, 0.7, 0.7)})]
        out = paired_differences(
            flatten("full", recs),
            field="order",
            plus="fixed",
            minus="bic",
            match=("task_id", "method", "inverse", "stage"),
            group_by=("stage",),
        )
        assert math.isnan(out[("roi",)]["sd"]) and out[("roi",)]["n"] == 1

    def test_the_direction_accuracy_can_be_the_measure(self):
        recs = [record(0, 0.0, {"gc/bic": (1, 1, 1)})]
        rows = flatten("full", recs)
        for r in rows:
            r["direction_accuracy"] = 0.5
        assert rows[0]["direction_accuracy"] == 0.5  # kept in the flat rows


class TestRuntime:
    def published(self):
        return {
            "full": (
                header("full"),
                [record(0, 0.0, {"gc/bic": (1, 1, 1)})],
                {"seconds": 1224.0, "n_failed": 0},
            ),
            "noise": (
                header("noise", gate={"estimate_hours": 0.04, "workers": 6}),
                [],
                {"seconds": 108.0},
            ),
        }

    def test_one_row_per_protocol_with_estimate_measured_and_workers(self):
        text = runtime_markdown(self.published())
        assert "| full | 1 | 0.40 | 0.34 | 6 |" in text  # 1224 s = 0.34 h
        assert "| noise | 0 | 0.04 | 0.03 | 6 |" in text

    def test_the_total_is_compared_with_the_two_hours(self):
        text = runtime_markdown(self.published())
        assert "total" in text.lower() and "0.37" in text and "2 h" in text

    def test_a_protocol_without_a_summary_line_says_so_instead_of_inventing_a_time(self):
        published = {"full": (header("full"), [], None)}
        assert "unknown" in runtime_markdown(published)

    def test_versions_that_differ_between_files_are_flagged(self):
        other = header("noise")
        other["environment"] = {**ENV, "numpy": "2.5.0"}
        published = {
            "full": (header("full"), [], {"seconds": 10.0}),
            "noise": (other, [], {"seconds": 10.0}),
        }
        text = runtime_markdown(published)
        assert "differ" in text and "numpy" in text

    def test_equal_versions_are_stated_once(self):
        text = runtime_markdown(self.published())
        assert "numpy 2.4.6" in text and "differ" not in text

    def test_an_add_on_is_listed_apart_and_not_in_the_total(self):
        published = {
            "full": (header("full"), [], {"seconds": 3600.0}),
            "noise_hard": (
                header("noise_hard", gate={"estimate_hours": 0.05, "workers": 6}),
                [],
                {"seconds": 360.0},
            ),
        }
        text = runtime_markdown(published)
        assert "| **total** | | | **1.00** |" in text  # 3600 s only, not 3960 s
        assert "Add-on protocols" in text
        add_on_line = "| noise_hard | 0 | 0.05 | 0.10 | 6 |"
        assert add_on_line in text
        assert text.index("**total**") < text.index("Add-on protocols") < text.index(add_on_line)

    def test_without_an_add_on_there_is_no_add_on_part(self):
        assert "Add-on" not in runtime_markdown(self.published())

    def test_the_code_line_still_covers_the_add_ons(self):
        published = {
            "full": (
                header("full", code={"git_commit": "a" * 40, "git_dirty": False}),
                [],
                {"seconds": 1.0},
            ),
            "noise_hard": (
                header("noise_hard", code={"git_commit": "b" * 40, "git_dirty": False}),
                [],
                {"seconds": 1.0},
            ),
        }
        after = runtime_markdown(published).split("Code")[1]
        assert "differ" in after and "noise_hard bbbbbbb" in after


class TestLoadAndBuild:
    def test_a_missing_file_is_an_error_naming_the_protocols(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="full"):
            load_published(tmp_path)

    def test_the_published_protocols_are_the_ten_of_the_plan(self):
        assert set(PUBLISHED) == {
            "full",
            "order",
            "inverse",
            "coupling",
            "extent",
            "size",
            "leakage_adjacent",
            "leakage_pairs",
            "significance",
            "noise",
        }

    def test_the_add_ons_are_four_named_protocols_apart_from_the_published_ten(self):
        assert ADDONS == (
            "noise_hard",
            "significance_power",
            "m5_methods",
            "asymptotic_calibration",
        )
        assert not set(ADDONS) & set(PUBLISHED)

    def test_the_further_methods_add_on_gets_auroc_and_direction_grids_for_every_stage(self):
        full = (header("full"), [record(0, 0.0, {"gc/bic": (1, 1, 1)})], {"seconds": 1.0})
        assert "Further methods" not in build_report({"full": full})
        m5 = (header("m5_methods"), [record(0, 0.0, {"gc/bic": (1, 1, 1)})], {"seconds": 1.0})
        text = build_report({"full": full, "m5_methods": m5})
        section = text[text.index("## Further methods on the head models (add-on)") :]
        assert section.count("Stage: roi.") == 2 and section.count("Stage: source.") == 2
        assert "Mean direction accuracy" in section

    def test_an_add_on_that_ran_gets_a_section_and_one_that_did_not_is_not_mentioned(self):
        full = (header("full"), [record(0, 0.0, {"gc/bic": (1, 1, 1)})], {"seconds": 1.0})
        without = build_report({"full": full})
        assert "noise_hard" not in without and "significance_power" not in without
        add_on = (header("noise_hard"), [], {"seconds": 1.0})
        with_it = build_report({"full": full, "noise_hard": add_on})
        assert "## Noise: harder SNR (add-on)" in with_it
        assert "significance_power" not in with_it

    def test_the_noise_table_is_drawn_for_the_harder_snr_too(self):
        records = []
        for noise in ("simulated", "white"):
            for seed in range(3):
                r = record(seed, -15.0, {"gc/bic": (0.8, 0.7 if noise == "white" else 0.9, 1.0)})
                if noise != "simulated":
                    r["task"]["noise"] = noise
                records.append(r)
        text = build_report(
            {
                "full": (header("full"), [], {"seconds": 1.0}),
                "noise_hard": (header("noise_hard"), records, {"seconds": 1.0}),
            }
        )
        section = text.split("## Noise: harder SNR (add-on)")[1]
        assert "| modality | SNR | estimator | stage |" in section and "-15" in section

    def test_the_report_has_the_conclusions_first_and_a_section_per_question(self):
        published = {
            "full": (
                header("full"),
                [
                    record(
                        s,
                        snr,
                        {
                            "gc/bic": (1, 0.9, 0.8),
                            "gc/fixed": (1, 0.95, 0.85),
                            "dtf/bic": (1, 0.8, 0.7),
                        },
                    )
                    for s in range(3)
                    for snr in (0.0, -10.0)
                ],
                {"seconds": 100.0},
            )
        }
        text = build_report(published, conclusions="# Conclusions\n\nA sentence.\n")
        assert text.startswith("# Conclusions")
        for heading in (
            "## Run time",
            "## Recovery by stage",
            "## Order setting",
            "## Conditional against pairwise",
        ):
            assert heading in text
        assert text.index("# Conclusions") < text.index("## Run time")

    def test_it_is_deterministic(self):
        published = {
            "full": (header("full"), [record(0, 0.0, {"gc/bic": (1, 1, 1)})], {"seconds": 1.0})
        }
        assert build_report(published) == build_report(published)

    def test_a_protocol_that_was_not_run_is_named_not_silently_missing(self):
        text = build_report(
            {"full": (header("full"), [record(0, 0.0, {"gc/bic": (1, 1, 1)})], {"seconds": 1.0})}
        )
        assert "not run" in text.lower() and "order" in text.lower()


class TestCodeInTheRunTimeSection:
    def published(self, commits, dirty=False):
        out = {}
        for i, commit in enumerate(commits):
            h = header(f"p{i}", code={"git_commit": commit, "git_dirty": dirty})
            out[f"p{i}"] = (h, [], {"seconds": 10.0})
        return out

    def test_one_commit_is_stated_once_in_short_form(self):
        text = runtime_markdown(self.published(["abcdef1234567890" * 2 + "abcdef12"] * 2))
        assert "git abcdef1" in text and "differ" not in text.split("Code")[1]

    def test_commits_that_differ_between_the_files_are_flagged(self):
        text = runtime_markdown(self.published(["a" * 40, "b" * 40]))
        assert "Code" in text and "differ" in text.split("Code")[1]

    def test_an_unclean_tree_is_flagged(self):
        assert "uncommitted" in runtime_markdown(self.published(["a" * 40], dirty=True)).lower()

    def test_a_file_without_a_code_entry_is_said_to_be_unknown(self):
        assert (
            "unknown"
            in runtime_markdown({"full": (header("full"), [], {"seconds": 1.0})}).split("Code")[1]
        )


def test_the_gitignore_lets_exactly_the_published_and_add_on_results_through():
    lines = [x.strip() for x in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    allowed = {
        x.removeprefix("!benchmark_results/").removesuffix(".jsonl")
        for x in lines
        if x.startswith("!benchmark_results/")
    }
    assert allowed == set(PUBLISHED) | set(ADDONS)
    assert "benchmark_results/*" in lines and "quick" not in allowed
