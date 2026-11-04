"""scripts/make_report.py: writes the generated report from a directory of results files."""

import importlib.util
import json
from pathlib import Path

import pytest

from dirconn.report import PUBLISHED, load_add_ons

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "make_report", ROOT / "scripts" / "make_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


def write_results(directory: Path, name: str) -> None:
    card = {"auroc": 0.9, "direction_accuracy": 1.0, "separation": 2.0}
    header = {
        "type": "header",
        "kind": "dirconn-benchmark",
        "version": 1,
        "protocol": name,
        "settings": {"estimators": ["gc/bic"]},
        "environment": {"python": "3.11.16", "numpy": "2.4.6"},
        "gate": {"estimate_hours": 0.1, "workers": 6},
    }
    record = {
        "type": "task",
        "status": "ok",
        "id": f"{name}-0",
        "task": {
            "scenario": "chain",
            "snr_db": 0.0,
            "duration_s": 30.0,
            "coupling": None,
            "source_model": "partial",
            "modality": "eeg",
            "seed_index": 0,
        },
        "result": {"gc/bic": {"roi": card, "sensor": card, "source": card, "agreement": {}}},
    }
    summary = {"type": "summary", "n_tasks": 1, "n_done": 1, "n_failed": 0, "seconds": 360.0}
    lines = [json.dumps(x) for x in (header, record, summary)]
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def results(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    for name in PUBLISHED:
        write_results(directory, name)
    return directory


def run(results, tmp_path, *extra):
    output = tmp_path / "report.md"
    code = script.main(
        [
            "--results-dir",
            str(results),
            "--output",
            str(output),
            "--conclusions",
            str(tmp_path / "conclusions.md"),
            *extra,
        ]
    )
    return code, output


class TestMakeReport:
    def test_it_writes_the_generated_report(self, results, tmp_path, capsys):
        code, output = run(results, tmp_path)
        assert code == 0
        text = output.read_text(encoding="utf-8")
        assert "# Benchmark results (generated)" in text and "## Run time" in text
        assert "10 of 10 protocols" in capsys.readouterr().out
        assert "Not run" not in text

    def test_the_hand_written_conclusions_come_first_and_unchanged(self, results, tmp_path):
        (tmp_path / "conclusions.md").write_text(
            "# Conclusions\n\nAn observed sentence.\n", encoding="utf-8"
        )
        _, output = run(results, tmp_path)
        text = output.read_text(encoding="utf-8")
        assert text.startswith("# Conclusions\n\nAn observed sentence.")

    def test_a_missing_results_file_is_an_error_naming_the_protocols(
        self, results, tmp_path, capsys
    ):
        (results / "order.jsonl").unlink()
        (results / "noise.jsonl").unlink()
        code, output = run(results, tmp_path)
        err = capsys.readouterr().err
        assert code == 2 and not output.exists()
        assert "order" in err and "noise" in err and "--allow-missing" in err

    def test_allow_missing_builds_the_rest_and_names_what_was_not_run(
        self, results, tmp_path, capsys
    ):
        (results / "order.jsonl").unlink()
        code, output = run(results, tmp_path, "--allow-missing")
        assert code == 0
        assert "Not run, so absent from this report: order" in output.read_text(encoding="utf-8")
        assert "9 of 10 protocols" in capsys.readouterr().out

    def test_the_same_results_give_the_same_report(self, results, tmp_path):
        _, first = run(results, tmp_path)
        text = first.read_text(encoding="utf-8")
        _, second = run(results, tmp_path)
        assert second.read_text(encoding="utf-8") == text


class TestAddOns:
    def test_add_ons_are_loaded_when_their_file_exists_and_never_required(self, results):
        assert load_add_ons(results) == {}
        write_results(results, "noise_hard")
        assert set(load_add_ons(results)) == {"noise_hard"}

    def test_an_add_on_file_is_included_apart_from_the_ten(self, results, tmp_path, capsys):
        write_results(results, "noise_hard")
        code, output = run(results, tmp_path)
        text = output.read_text(encoding="utf-8")
        assert code == 0
        assert "10 of 10 protocols, plus 1 add-on" in capsys.readouterr().out
        assert "Add-on protocols" in text and "| noise_hard |" in text

    def test_a_missing_add_on_is_neither_an_error_nor_mentioned(self, results, tmp_path):
        code, output = run(results, tmp_path)
        text = output.read_text(encoding="utf-8")
        assert code == 0 and "noise_hard" not in text and "significance_power" not in text

    def test_an_add_on_does_not_rescue_a_missing_published_protocol(self, results, tmp_path):
        write_results(results, "noise_hard")
        (results / "order.jsonl").unlink()
        code, output = run(results, tmp_path)
        assert code == 2 and not output.exists()
