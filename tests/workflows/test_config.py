"""Tests for the typed pipeline configuration."""

from __future__ import annotations

import pydantic
import pytest

from dirconn.stats.correction import METHODS as CORRECTIONS
from dirconn.stats.significance import _NAMED_SURROGATES
from dirconn.workflows.config import (
    CORRECTION_METHODS,
    SURROGATES,
    EstimationConfig,
    PipelineConfig,
    PrepareConfig,
    StatisticsConfig,
    analysis_hash,
    apply_overrides,
    config_hash,
    dump_config,
    load_config,
)


def test_default_config_loads_and_hashes():
    cfg = load_config(None)
    assert cfg == PipelineConfig()
    assert len(config_hash(cfg)) == 16
    assert config_hash(cfg) == config_hash(PipelineConfig())


def test_surrogate_and_correction_names_match_the_library():
    assert set(SURROGATES) == set(_NAMED_SURROGATES)
    assert tuple(CORRECTION_METHODS) == tuple(CORRECTIONS)
    StatisticsConfig(surrogate="aaft", correction="bonferroni")
    for bad in ({"surrogate": "block_shuffle"}, {"correction": "holm"}):
        with pytest.raises(pydantic.ValidationError):
            StatisticsConfig(**bad)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[estimation]\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_config(path)
    with pytest.raises(pydantic.ValidationError):
        PipelineConfig.model_validate({"nonsense": {}})


def test_unknown_method_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="nope"):
        EstimationConfig(methods=("gc", "nope"))


def test_params_for_a_method_not_in_methods_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="params"):
        EstimationConfig(methods=("gc",), params={"rpdc": {"x": 1}})


def test_band_above_nyquist_is_rejected():
    with pytest.raises(pydantic.ValidationError, match="Nyquist") as exc:
        PipelineConfig(
            prepare=PrepareConfig(resample_hz=250.0),
            estimation=EstimationConfig(bands={"gamma": (80.0, 200.0)}),
        )
    assert "gamma" in str(exc.value)
    assert "200" in str(exc.value)
    assert "125" in str(exc.value)


def test_bad_bands_and_schema_version_are_rejected():
    for bands in ({}, {"x": (8.0, 4.0)}, {"x": (-1.0, 4.0)}):
        with pytest.raises(pydantic.ValidationError):
            EstimationConfig(bands=bands)
    with pytest.raises(pydantic.ValidationError):
        PipelineConfig(schema_version=2)


def test_config_is_frozen():
    cfg = PipelineConfig()
    with pytest.raises(pydantic.ValidationError):
        cfg.schema_version = 2  # type: ignore[misc]


def test_override_changes_nested_field():
    cfg = apply_overrides(
        PipelineConfig(), ["estimation.methods=gc,rpdc", "statistics.n_surrogates=20"]
    )
    assert cfg.estimation.methods == ("gc", "rpdc")
    assert cfg.statistics.n_surrogates == 20
    assert PipelineConfig().statistics.n_surrogates == 200


def test_override_types_none_bool_and_params():
    cfg = apply_overrides(
        PipelineConfig(),
        [
            "prepare.resample_hz=null",
            "prepare.zscore=false",
            "estimation.order=aic",
            "estimation.params.gc.method=state_space",
            "estimation.params.gc.n=3",
            "prepare.lock.events=a,b",
        ],
    )
    assert cfg.prepare.resample_hz is None
    assert cfg.prepare.zscore is False
    assert cfg.estimation.order == "aic"
    assert cfg.estimation.params == {"gc": {"method": "state_space", "n": 3}}
    assert cfg.prepare.lock.events == ("a", "b")


def test_override_unknown_path_or_bad_form_raises_value_error():
    with pytest.raises(ValueError, match="estimation.nope"):
        apply_overrides(PipelineConfig(), ["estimation.nope=1"])
    with pytest.raises(ValueError, match="no_equals"):
        apply_overrides(PipelineConfig(), ["no_equals"])
    with pytest.raises(ValueError):
        apply_overrides(PipelineConfig(), ["statistics.n_surrogates=many"])


def test_dump_then_load_round_trips(tmp_path):
    cfg = apply_overrides(
        PipelineConfig(),
        ["estimation.methods=gc,te", "estimation.order=7", "prepare.lock.events=a,b"],
    )
    path = tmp_path / "out.toml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    assert config_hash(loaded) == config_hash(cfg)


def test_estimation_params_subtable_round_trips(tmp_path):
    path = tmp_path / "in.toml"
    path.write_text(
        '[estimation]\nmethods = ["gc", "rpdc"]\n\n[estimation.gc]\nmethod = "state_space"\n'
        "\n[estimation.rpdc]\nalpha = 0.5\nflag = true\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.estimation.params == {
        "gc": {"method": "state_space"},
        "rpdc": {"alpha": 0.5, "flag": True},
    }
    out = tmp_path / "out.toml"
    dump_config(cfg, out)
    assert "[estimation.gc]" in out.read_text(encoding="utf-8")
    assert load_config(out) == cfg
    assert config_hash(load_config(out)) == config_hash(cfg)


def test_analysis_hash_ignores_output_and_execution():
    base = PipelineConfig()
    other = apply_overrides(base, ["output.long_table=false", "execution.seed=1"])
    assert analysis_hash(base) == analysis_hash(other)
    assert config_hash(base) != config_hash(other)
    changed = apply_overrides(base, ["statistics.alpha=0.01"])
    assert analysis_hash(base) != analysis_hash(changed)
    input_changed = apply_overrides(base, ["input.level=sensor"])
    assert analysis_hash(base) == analysis_hash(input_changed)


def test_none_fields_round_trip_through_dump(tmp_path):
    cfg = apply_overrides(PipelineConfig(), ["prepare.resample_hz=null", "input.description=null"])
    assert cfg.prepare.resample_hz is None
    assert cfg.input.description is None
    path = tmp_path / "out.toml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    assert config_hash(loaded) == config_hash(cfg)


def test_null_sentinel_is_accepted_in_toml(tmp_path):
    path = tmp_path / "in.toml"
    path.write_text(
        '[prepare]\nresample_hz = "NULL"\n\n[input]\ndescription = "null"\n', encoding="utf-8"
    )
    cfg = load_config(path)
    assert cfg.prepare.resample_hz is None
    assert cfg.input.description is None


def test_keys_with_spaces_and_dots_round_trip(tmp_path):
    cfg = PipelineConfig(
        prepare=PrepareConfig(resample_hz=250.0),
        estimation=EstimationConfig(
            bands={"beta low": (13.0, 20.0), "beta.low": (20.0, 30.0), "muμ": (8.0, 12.0)},
            params={"gc": {"odd.key": 1, "with space": "x"}},
        ),
    )
    path = tmp_path / "out.toml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    assert config_hash(loaded) == config_hash(cfg)


@pytest.mark.parametrize(
    "override",
    [
        "statistics.alpha=0",
        "statistics.alpha=1",
        "statistics.n_surrogates=0",
        "estimation.order=0",
        "estimation.order_max=0",
        "prepare.lock.tmin=2",
        "prepare.pad_bad_seconds=-1",
        "prepare.resample_hz=0",
        "prepare.resample_hz=-5",
    ],
)
def test_out_of_range_values_are_rejected(override):
    with pytest.raises(pydantic.ValidationError):
        apply_overrides(PipelineConfig(), [override])


def test_bad_exclude_freqs_are_rejected():
    for ranges in (((50.0, 40.0),), ((-1.0, 5.0),), ((5.0, 5.0),)):
        with pytest.raises(pydantic.ValidationError):
            EstimationConfig(exclude_freqs=ranges)
    EstimationConfig(exclude_freqs=((49.0, 51.0),))
