"""TOML configurations of the session pipeline fixtures (``tests/conftest.py``)."""

from __future__ import annotations

PIPELINE_CONFIG = """\
[prepare]
resample_hz = "null"

[estimation]
methods = ["gc", "pdc", "adtf"]
order = "bic"
order_max = 6

[controls]
time_reversal = true

[statistics]
n_surrogates = 19
"""

FAILURE_CONFIG = """\
[prepare]
resample_hz = "null"

[estimation]
methods = ["gc", "pdc"]
order = "bic"
order_max = 6

[estimation.gc]
order = 100000

[controls]
time_reversal = true

[statistics]
n_surrogates = 19
"""

EPOCHED_CONFIG = """\
[input]
level = "sensor"
datatypes = ["eeg"]

[prepare]
resample_hz = "null"

[prepare.lock]
events = ["cue"]
tmin = -0.5
tmax = 1.0

[estimation]
methods = ["gc", "wdtf"]
order = 2

[estimation.wdtf]
window = 0.5
step = 0.25

[controls]
time_reversal = true

[statistics]
n_surrogates = 5
"""
