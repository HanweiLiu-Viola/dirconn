import json
import warnings

import numpy as np

from dirconn import TimeSeries
from dirconn.workflows.config import PipelineConfig
from dirconn.workflows.steps.diagnose import diagnose

SFREQ = 250.0


def _ts(valid=None):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, int(20 * SFREQ))) * [[1.0], [3.0]]
    return TimeSeries.from_array(data, SFREQ, valid=valid)


def _cfg(spectral=True):
    return PipelineConfig.model_validate({"controls": {"spectral_range_check": spectral}})


def test_diagnostics_to_dict_is_json_serialisable():
    valid = np.ones(int(20 * SFREQ), dtype=bool)
    valid[:500] = False
    diag = diagnose(_ts(valid), _cfg())
    names = _ts().names
    assert diag.valid_fraction == 1 - 500 / valid.size
    assert diag.n_trials == 1
    assert diag.n_times == valid.size
    assert diag.sfreq == SFREQ
    assert diag.channel_std[names[1]] > 2 * diag.channel_std[names[0]]
    back = json.loads(json.dumps(diag.to_dict(), allow_nan=False))
    assert back["checks"][0]["name"] == "spectral_dynamic_range"
    assert back["spectral"]["worst_channel"] in names


def test_spectral_check_can_be_switched_off():
    diag = diagnose(_ts(), _cfg(spectral=False))
    assert diag.spectral is None
    assert diag.checks == ()
    assert diag.valid_fraction == 1.0
    assert diag.to_dict()["spectral"] is None


def test_flat_channel_diagnostics_stay_json_safe():
    data = np.array(_ts().data[0])
    data[0] = 2.0
    flat = TimeSeries.from_array(data, SFREQ)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diag = diagnose(flat, _cfg())
    assert diag.checks[0].status == "warn"
    json.dumps(diag.to_dict(), allow_nan=False)


def test_channel_without_valid_samples_has_none_std():
    valid = np.zeros(int(20 * SFREQ), dtype=bool)
    diag = diagnose(_ts(valid), _cfg(spectral=False))
    assert all(v is None for v in diag.channel_std.values())
    json.dumps(diag.to_dict(), allow_nan=False)
