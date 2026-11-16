import json

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
