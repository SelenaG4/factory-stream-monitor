"""Synthesize labeled Swiss-factory telemetry.

Healthy readings are drawn from each sensor's nominal envelope; a fraction of events
carry an injected fault whose signature nudges a physically-sensible subset of sensors
(a worn bearing raises vibration + acoustic + heat, an overheating spindle spikes
temperature + load + power, ...). The fault flag is the ground-truth label the MLlib
model learns and the evaluation scores against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stream.config import (ANOMALY_RATE, FAULTS, MACHINE_TYPES, SENSOR_SPECS,
                           SENSORS, SITES)


def _machines(per_site=4):
    out = []
    for site in SITES:
        for i in range(per_site):
            mt = MACHINE_TYPES[i % len(MACHINE_TYPES)]
            out.append((site, f"{site[:3].upper()}-{mt.split('_')[0][:2].upper()}-{i:02d}", mt))
    return out


def generate(n_events: int, start="2018-06-01", freq_seconds=2, seed=42,
             anomaly_rate=ANOMALY_RATE, drift_start_frac=None, drift=None) -> pd.DataFrame:
    """If drift_start_frac/drift are set, the tail of the stream gets a *distribution*
    shift (a slow ramp added to the given sensors) with the labels left unchanged --
    simulating fleet-wide degradation (e.g. a hot spell / coolant loss) that the drift
    monitor should catch even though it isn't a labeled point anomaly."""
    rng = np.random.default_rng(seed)
    machines = _machines()
    times = pd.date_range(start, periods=n_events, freq=f"{freq_seconds}s")
    mi = rng.integers(0, len(machines), n_events)

    rows = {s: np.empty(n_events) for s in SENSORS}
    for s, spec in SENSOR_SPECS.items():
        rows[s] = rng.normal(spec["mean"], spec["std"], n_events)

    labels = (rng.random(n_events) < anomaly_rate).astype(int)
    fault_names = np.array(list(FAULTS))
    faults = np.where(labels == 1, rng.choice(fault_names, n_events), None)

    for i in np.flatnonzero(labels):
        sig = FAULTS[faults[i]]
        for s, delta in sig.items():
            # scale the nudge a bit so faults vary in severity (0.6x..1.4x)
            rows[s][i] += delta * (0.6 + 0.8 * rng.random())

    # optional fleet-wide distribution drift on the tail of the stream (unlabeled)
    if drift_start_frac is not None and drift:
        i0 = int(n_events * drift_start_frac)
        ramp = np.zeros(n_events)
        ramp[i0:] = np.linspace(0, 1, n_events - i0)
        for s, delta in drift.items():
            rows[s] = rows[s] + ramp * delta

    # clip to physical bounds
    for s, spec in SENSOR_SPECS.items():
        rows[s] = np.clip(rows[s], spec["lo"], spec["hi"])

    df = pd.DataFrame({
        "event_id": [f"e{ix:08d}" for ix in range(n_events)],
        "event_time": times,
        "site": [machines[m][0] for m in mi],
        "machine_id": [machines[m][1] for m in mi],
        "machine_type": [machines[m][2] for m in mi],
        **{s: np.round(rows[s], 3) for s in SENSORS},
        "label": labels,
        "fault_type": faults,
    })
    return df
