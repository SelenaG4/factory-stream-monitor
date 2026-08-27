import numpy as np

from stream.generate import generate


def test_label_rate_matches_target(sample_events):
    assert abs(sample_events["label"].mean() - 0.06) < 0.02
    # faulty events name a fault; healthy ones don't
    assert sample_events.loc[sample_events.label == 1, "fault_type"].notna().all()
    assert sample_events.loc[sample_events.label == 0, "fault_type"].isna().all()


def test_faults_elevate_the_right_sensors(sample_events):
    healthy = sample_events[sample_events.label == 0]
    faulty = sample_events[sample_events.label == 1]
    for s in ["temperature_c", "vibration_mm_s", "power_kw"]:
        assert faulty[s].mean() > healthy[s].mean()


def test_drift_injection_shifts_only_the_tail():
    df = generate(4000, seed=1, drift_start_frac=0.5,
                  drift={"temperature_c": 20.0})
    head = df.iloc[:2000]["temperature_c"].mean()
    tail = df.iloc[-500:]["temperature_c"].mean()
    assert tail > head + 5          # tail clearly drifted up
    # labels unchanged by drift (drift is a distribution shift, not a point anomaly)
    assert abs(df["label"].mean() - 0.06) < 0.02
