"""Dockerfile and .dockerignore: dirconn-bids is the entrypoint and the image records its tag.

The static tests only read the files. ``test_image_runs_the_fixture`` needs a Docker daemon and a
built image (``DIRCONN_TEST_IMAGE=<tag>``) and runs only with ``-m docker``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def _joined() -> str:
    """The Dockerfile with line continuations (backslash, end of line) joined into one line."""
    return re.sub(r"\\[ \t]*\r?\n", " ", _dockerfile())


def _instructions(name: str) -> list[str]:
    """Return the arguments of every ``name`` instruction (line continuations joined)."""
    text = _joined()
    return [
        m.group(1).strip()
        for m in re.finditer(rf"^{name}\s+(.*)$", text, flags=re.MULTILINE | re.IGNORECASE)
    ]


def test_dockerfile_entrypoint_is_the_cli():
    assert [json.loads(x) for x in _instructions("ENTRYPOINT")] == [["dirconn-bids"]]
    assert [json.loads(x) for x in _instructions("CMD")] == [["--help"]]
    assert 'CMD ["bash"]' not in _dockerfile()
    assert "EXPOSE 8888" in _dockerfile()


def test_dockerfile_installs_the_app_extra():
    text = _joined()
    match = re.search(r"/opt/dirconn-src\[([^\]]*)\]", text)
    assert match is not None
    extras = set(match.group(1).split(","))
    assert {"connectivity", "pcmci", "stats", "parallel", "speed", "plot", "app"} == extras
    smoke = re.search(r'python -c "(import [^"]*)"', text)
    assert smoke is not None
    for name in ("jinja2", "pydantic", "dirconn.reports.collect"):
        assert name in smoke.group(1)


def test_dockerfile_sets_the_provenance_variables():
    lines = _joined().split("\n")
    last_arg = max(i for i, x in enumerate(lines) if x.startswith("ARG "))
    env = " ".join(x for x in lines[last_arg:] if x.startswith("ENV "))
    assert "DIRCONN_IMAGE_TAG=${IMAGE}" in env
    assert "DIRCONN_BUILD_REVISION=${GIT_SHA}" in env
    for kept in ("DIRCONN_GIT_COMMIT=${GIT_SHA}", "DIRCONN_GIT_DIRTY=${GIT_DIRTY}"):
        assert kept in env
    assert "DIRCONN_IMAGE=${IMAGE}" in env


def test_dockerignore_excludes_agent_and_private_dirs():
    entries = {
        x.strip()
        for x in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.startswith("#")
    }
    assert {".superpowers/", ".claude/", "_notes/", ".git"} <= entries


@pytest.mark.docker
def test_image_runs_the_fixture(tmp_path: Path):
    import os

    image = os.environ.get("DIRCONN_TEST_IMAGE")
    if not image:
        pytest.skip("set DIRCONN_TEST_IMAGE to the built image tag to run this test")
    if shutil.which("docker") is None:
        pytest.skip("docker is not on PATH")
    inspect = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if inspect.returncode != 0:
        pytest.skip(f"docker image {image} is not available locally")

    bids = tmp_path / "bids"
    shutil.copytree(ROOT / "tests" / "fixtures" / "bids_tiny", bids)
    out = tmp_path / "out"
    out.mkdir()
    # The container runs as UID 1000; on Docker Desktop for Windows bind mounts are writable. On
    # Linux with another host UID add `--user $(id -u):$(id -g)` to the command.
    command = [
        "docker", "run", "--rm", "-v", f"{bids}:/data:ro", "-v", f"{out}:/out", image,
        "/data", "/out", "participant",
        "--set", "prepare.resample_hz=null",
        "--set", "estimation.methods=gc",
        "--set", "estimation.order=3",
        "--set", "statistics.n_surrogates=19",
        "--skip-bids-validation",
    ]  # fmt: skip
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert (out / "dirconn" / "sub-01.html").is_file()
    run_dir = sorted((out / "dirconn" / "logs").glob("run-*"))[-1]
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["image"]["tag"]
    lock = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, "/opt/environment-lock.txt"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    kept = (run_dir / "environment-lock.txt").read_text(encoding="utf-8")
    assert kept.replace("\r\n", "\n") == lock.replace("\r\n", "\n")
