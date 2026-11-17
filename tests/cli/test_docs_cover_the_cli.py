"""The command-line and configuration pages document every option and every field."""

from __future__ import annotations

import typing
from pathlib import Path

from pydantic import BaseModel

from dirconn.cli.parser import SUBCOMMANDS, build_parser
from dirconn.workflows.config import PipelineConfig

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _option_strings() -> list[str]:
    return [opt for action in build_parser()._actions for opt in action.option_strings]


def _field_paths(model: type[BaseModel], prefix: str = "") -> list[tuple[str, bool]]:
    """Dotted paths of every field, with whether the field holds a dict (keys are free)."""
    paths: list[tuple[str, bool]] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths.extend(_field_paths(annotation, f"{path}."))
        else:
            paths.append((path, typing.get_origin(annotation) is dict))
    return paths


def test_docs_mention_every_option():
    text = (DOCS / "usage.rst").read_text(encoding="utf-8")
    options = _option_strings()
    assert "--participant-label" in options  # the parser was read
    missing = [opt for opt in options if f"``{opt}" not in text]
    assert not missing, f"options not documented in docs/usage.rst: {missing}"
    missing_commands = [cmd for cmd in SUBCOMMANDS if f"``{cmd}" not in text]
    assert not missing_commands, missing_commands


def test_the_default_config_shown_is_the_packaged_one():
    # the page shows a copy (the docs are built from a copy of docs/, so no literalinclude)
    text = (DOCS / "configuration.rst").read_text(encoding="utf-8")
    default = DOCS.parent / "src" / "dirconn" / "workflows" / "_data" / "default_config.toml"
    block = "".join(
        f"   {line}\n" if line.strip() else "\n"
        for line in default.read_text(encoding="utf-8").splitlines()
    )
    assert ".. code-block:: toml\n\n" + block in text


def test_every_config_field_is_documented():
    text = (DOCS / "configuration.rst").read_text(encoding="utf-8")
    paths = _field_paths(PipelineConfig)
    assert ("prepare.lock.events", False) in paths  # nested sections are walked
    missing = []
    for path, is_dict in paths:
        # a dict-valued field is documented by its key pattern, e.g. estimation.bands.<name>
        needle = f"``{path}.<" if is_dict else f"``{path}``"
        if needle not in text:
            missing.append(needle)
    assert not missing, f"fields not documented in docs/configuration.rst: {missing}"
