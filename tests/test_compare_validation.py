"""scripts/compare_validation.py: two validation runs, row by row and by multi-seed medians."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "compare_validation", ROOT / "scripts" / "compare_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
META = {"n_times": 100, "seeds": [0, 1, 2], "sfreq": 250.0, "spectral_band": [1, 80]}


def run(tmp_path, name, aurocs, meta=META):
    rows = [
        {"scenario": "chain", "method": "gc", "seed": s, "auroc": a, "auprc": 1.0,
         "direction_accuracy": 1.0, "separation": 2.0, "order": 2}
        for s, a in enumerate(aurocs)
    ]  # fmt: skip
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, "rows": rows}), encoding="utf-8")
    return path


def test_identical_runs_agree_exactly(tmp_path):
    code, report = script.compare(
        run(tmp_path, "a", [1, 0.9, 0.8]), run(tmp_path, "b", [1, 0.9, 0.8])
    )
    assert code == 0 and "3 of 3 rows agree exactly" in report and "No median" in report


def test_a_changed_median_is_listed(tmp_path):
    code, report = script.compare(
        run(tmp_path, "a", [1, 0.9, 0.8]), run(tmp_path, "b", [1, 0.7, 0.8])
    )
    assert code == 0 and "2 of 3 rows" in report and "| chain | gc | auroc | 0.9 | 0.8 |" in report


def test_a_change_that_keeps_the_median_is_not_listed(tmp_path):
    code, report = script.compare(
        run(tmp_path, "a", [1, 0.9, 0.8]), run(tmp_path, "b", [1, 0.9, 0.7])
    )
    assert "No median" in report and "0.1" in report


def test_runs_with_other_settings_are_refused(tmp_path):
    other = dict(META, n_times=200)
    code, _ = script.compare(run(tmp_path, "a", [1, 1, 1]), run(tmp_path, "b", [1, 1, 1], other))
    assert code == 1


def test_a_score_undefined_for_every_seed_is_not_a_difference(tmp_path, recwarn):
    nan = float("nan")
    code, report = script.compare(
        run(tmp_path, "a", [nan, nan, nan]), run(tmp_path, "b", [nan, nan, nan])
    )
    assert code == 0 and "No median" in report and not recwarn.list
