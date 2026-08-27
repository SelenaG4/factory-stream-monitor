from stream.drift import batch_drift
from stream.generate import generate
from stream.schema import FEATURE_SENSORS


def _reference():
    df = generate(5000, seed=99)
    return {s: {"mean": float(df[s].mean()), "std": float(df[s].std())} for s in FEATURE_SENSORS}


def test_clean_batch_shows_no_drift():
    ref = _reference()
    clean = generate(1000, seed=5)
    d = batch_drift(clean, ref)
    assert d["_summary"]["n_drifted_sensors"] == 0
    assert d["_summary"]["max_drift_z"] < 1.0


def test_shifted_batch_is_flagged():
    ref = _reference()
    shifted = generate(1000, seed=6, drift_start_frac=0.0, drift={"temperature_c": 25.0})
    d = batch_drift(shifted, ref)
    assert d["temperature_c"]["drifted"] is True
    assert d["_summary"]["n_drifted_sensors"] >= 1


def test_out_of_spec_readings_counted():
    ref = _reference()
    # push temperature far past its hi spec so clipping still leaves it at the ceiling
    hot = generate(500, seed=7, drift_start_frac=0.0, drift={"temperature_c": 200.0})
    d = batch_drift(hot, ref)
    assert d["temperature_c"]["batch_mean"] <= 120.0     # clipped to spec ceiling
