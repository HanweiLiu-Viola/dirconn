"""scripts/build_image.py: the image tag names the commit, and a dirty tree is marked."""

import importlib.util
from pathlib import Path

import pytest

# The script reads pyproject.toml with tomllib (Python 3.11+); images are built on the workstation.
pytest.importorskip("tomllib", reason="scripts/build_image.py needs Python 3.11 or later")

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "build_image", ROOT / "scripts" / "build_image.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
SHA = "0123456789abcdef0123456789abcdef01234567"


def test_the_tag_is_the_version_and_the_short_sha():
    assert script.image_tag("0.0.1.dev0", SHA, dirty=False) == "0.0.1.dev0-0123456"


def test_a_dirty_tree_is_marked():
    assert script.image_tag("0.0.1.dev0", SHA, dirty=True).endswith("-dirty")


def test_a_local_version_part_is_made_a_valid_docker_tag():
    assert "+" not in script.image_tag("0.1.0+local", SHA, dirty=False)


def test_the_build_passes_the_provenance_arguments():
    command = script.build_command("img:tag", SHA, "0.1", "2026-10-15T00:00:00Z", dirty=False)
    assert f"GIT_SHA={SHA}" in command and "VERSION=0.1" in command
    assert "GIT_DIRTY=0" in command and "IMAGE=img:tag" in command
    assert "BUILD_DATE=2026-10-15T00:00:00Z" in command and command[-1] == str(ROOT)


def test_a_push_from_an_uncommitted_tree_is_refused(monkeypatch):
    monkeypatch.setattr(script, "_git", lambda *a: SHA if a[0] == "rev-parse" else " M file")
    assert script.main(["--push", "--dry-run"]) == 1
