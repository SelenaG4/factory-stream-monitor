"""End-to-end local demo (no Kafka needed): generate a telemetry stream with a
developing fleet-wide drift in its tail, run the Spark streaming job over it, turn the
model's alerts into grounded incident reports, and write a compact summary the dashboard
reads. This is the offline pipeline that produces the committed demo artifacts, mirroring
the heavy-compute-offline / light-serving split used across this portfolio.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stream.config import STREAM_IN, STREAM_OUT  # noqa: E402
from stream.generate import generate  # noqa: E402
from stream.incident import incidents_from_alerts  # noqa: E402
from stream.producer import to_files  # noqa: E402
from stream.streaming_job import run  # noqa: E402

N_EVENTS = 9000


def main():
    import shutil
    shutil.rmtree(STREAM_IN, ignore_errors=True)

    # a stream that drifts in its final 40% (heat + vibration creeping up fleet-wide)
    df = generate(N_EVENTS, seed=7, drift_start_frac=0.6,
                  drift={"temperature_c": 14.0, "vibration_mm_s": 2.5})
    nfiles = to_files(df, batch_size=500)
    print(f"produced {len(df):,} events in {nfiles} files (drift injected in the tail)")

    spark, *_ = run(source="file")
    spark.stop()

    alerts = [json.loads(l) for l in open(STREAM_OUT / "alerts.jsonl")]
    drift = [json.loads(l) for l in open(STREAM_OUT / "drift.jsonl")]
    incidents = incidents_from_alerts(alerts)
    (STREAM_OUT / "incidents.json").write_text(json.dumps(incidents, indent=2))

    from collections import Counter
    summary = {
        "n_events": int(len(df)),
        "n_alerts": len(alerts),
        "alert_rate_pct": round(len(alerts) / len(df) * 100, 2),
        "true_anomaly_rate_pct": round(float(df["label"].mean()) * 100, 2),
        "n_incidents": len(incidents),
        "incidents_by_cause": dict(Counter(i["likely_cause"] for i in incidents)),
        "incidents_by_severity": dict(Counter(i["severity"] for i in incidents)),
        "sites": sorted(df["site"].unique().tolist()),
        "machines": int(df["machine_id"].nunique()),
        "drift_batches": len(drift),
        "max_drift_z_end": round(max((d["max_drift_z"] for d in drift[-3:]), default=0), 2),
        "drifted_at_end": int(drift[-1]["n_drifted_sensors"]) if drift else 0,
    }
    (STREAM_OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nartifacts in {STREAM_OUT}: alerts.jsonl, drift.jsonl, window_metrics/, "
          "incidents.json, run_summary.json")


if __name__ == "__main__":
    main()
