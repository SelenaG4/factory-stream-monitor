from stream.incident import KB, build_incident, incidents_from_alerts


def _alert(**over):
    base = dict(machine_id="ZUR-CN-00", site="Zurich", machine_type="CNC_Mill",
                temperature_c=95.0, vibration_mm_s=3.0, spindle_load_pct=90.0,
                power_kw=35.0, acoustic_db=80.0, anomaly_prob=0.95)
    base.update(over)
    return base


def test_incident_is_grounded_and_consistent():
    kb = KB()
    rep = build_incident(_alert(), kb)
    # cited source matches the stated cause (no cause/source disagreement)
    assert rep["likely_cause"].replace(" ", "_") in rep["source"]
    # action is quoted from a guide, not empty/invented
    assert len(rep["recommended_action"]) > 20
    assert rep["severity"] == "critical"          # prob 0.95


def test_severity_tracks_probability():
    kb = KB()
    assert build_incident(_alert(anomaly_prob=0.62), kb)["severity"] == "watch"
    assert build_incident(_alert(anomaly_prob=0.80), kb)["severity"] == "warning"
    assert build_incident(_alert(anomaly_prob=0.95), kb)["severity"] == "critical"


def test_overheating_signature_maps_to_spindle_guide():
    kb = KB()
    # very high temperature + load + power -> spindle overheating
    rep = build_incident(_alert(temperature_c=110.0, spindle_load_pct=98.0, power_kw=45.0,
                                vibration_mm_s=2.5, acoustic_db=78.0), kb)
    assert rep["likely_cause"] == "spindle overheat"
    assert "spindle_overheat" in rep["source"]


def test_batch_reports_are_ranked_by_probability():
    reps = incidents_from_alerts([_alert(anomaly_prob=0.7), _alert(anomaly_prob=0.99)], top_n=2)
    assert reps[0]["anomaly_prob"] >= reps[1]["anomaly_prob"]
