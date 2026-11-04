"""The version pins of the published benchmark numbers, and the deliberately empty CI pin."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(name):
    lines = (ROOT / "constraints" / name).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]


def test_the_benchmark_pins_are_exact_versions_of_the_libraries_the_numbers_depend_on():
    found = pins("benchmark.txt")
    assert found and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", p) for p in found)
    names = {p.split("==")[0].lower() for p in found}
    assert {"numpy", "scipy", "mne", "statsmodels", "joblib"} <= names


def test_the_ci_pin_is_empty_on_purpose_and_says_why():
    assert pins("ci.txt") == []
    assert "Intentionally empty" in (ROOT / "constraints" / "ci.txt").read_text(encoding="utf-8")


def test_the_contributing_guide_names_both_files_and_no_longer_says_ci_is_pinned():
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "constraints/benchmark.txt" in text and "constraints/ci.txt" in text
    assert "use the\n  pinned `constraints/ci.txt`" not in text


def test_the_docker_pins_are_exact_and_agree_with_the_benchmark_pins():
    docker = dict(p.lower().split("==") for p in pins("docker.txt"))
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", p) for p in pins("docker.txt"))
    assert "dirconn" not in docker
    for pin in pins("benchmark.txt"):
        name, version = pin.lower().split("==")
        assert docker.get(name) == version, name
