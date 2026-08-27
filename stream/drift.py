"""Data-quality + feature-drift monitoring on the live stream.

Two production concerns a model in production must watch, not just accuracy on a
held-out set from months ago:
  * data quality -- readings outside the physical spec band (a stuck/failed sensor);
  * feature drift -- the live input distribution moving away from what the model was
    trained on, which silently erodes accuracy even when nothing errors.

We compare each micro-batch's per-sensor mean to the training reference, expressed in
training standard deviations (a z-shift), and flag any sensor past a threshold.
"""
from __future__ import annotations

import json

from stream.config import REFERENCE_JSON, SENSOR_SPECS
from stream.schema import FEATURE_SENSORS

DRIFT_Z_THRESHOLD = 1.0     # |mean shift| in training std devs that counts as drift


def load_reference() -> dict:
    return json.loads(REFERENCE_JSON.read_text())


def batch_drift(pdf, reference: dict, z_threshold: float = DRIFT_Z_THRESHOLD) -> dict:
    """pdf: pandas frame of one micro-batch. Returns per-sensor drift + quality stats."""
    out = {}
    for s in FEATURE_SENSORS:
        col = pdf[s].astype(float)
        ref = reference[s]
        z = abs(col.mean() - ref["mean"]) / (ref["std"] if ref["std"] else 1.0)
        spec = SENSOR_SPECS[s]
        oob = int(((col < spec["lo"]) | (col > spec["hi"])).sum())
        out[s] = {
            "batch_mean": round(float(col.mean()), 3),
            "ref_mean": round(float(ref["mean"]), 3),
            "drift_z": round(float(z), 3),
            "drifted": bool(z > z_threshold),
            "out_of_spec": oob,
        }
    out["_summary"] = {
        "n_drifted_sensors": sum(1 for s in FEATURE_SENSORS if out[s]["drifted"]),
        "max_drift_z": round(max(out[s]["drift_z"] for s in FEATURE_SENSORS), 3),
        "total_out_of_spec": sum(out[s]["out_of_spec"] for s in FEATURE_SENSORS),
    }
    return out
