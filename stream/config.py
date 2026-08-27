"""Central configuration for the real-time factory monitor.

One place for the plant model (Swiss sites + machine types + sensor physics), the
Kafka/stream wiring, and the paths the offline model and the streaming job share.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_DIR = DATA / "model" / "anomaly_pipeline"       # persisted Spark MLlib PipelineModel
REFERENCE_JSON = DATA / "reference" / "feature_reference.json"  # training dist -> drift baseline
STREAM_IN = DATA / "stream_in"                         # file-source landing dir (sandbox demo)
STREAM_OUT = DATA / "stream_out"                       # alerts / window-metrics / drift sinks
CHECKPOINT = DATA / "_checkpoints"

# ---------------------------------------------------------------- plant model
SITES = ["Zurich", "Basel", "Geneva"]
MACHINE_TYPES = ["CNC_Mill", "Lathe", "Hydraulic_Press", "Surface_Grinder"]

# nominal operating envelope per sensor (mean, std) and a hard "spec" band used both
# to synthesize data and, later, as data-quality bounds in the streaming job
SENSOR_SPECS = {
    "temperature_c":   dict(mean=62.0, std=6.0,  lo=10.0, hi=120.0),
    "vibration_mm_s":  dict(mean=2.8,  std=0.7,  lo=0.0,  hi=20.0),
    "spindle_load_pct":dict(mean=68.0, std=12.0, lo=0.0,  hi=100.0),
    "power_kw":        dict(mean=22.0, std=5.0,  lo=0.0,  hi=80.0),
    "acoustic_db":     dict(mean=78.0, std=4.0,  lo=40.0, hi=120.0),
}
SENSORS = list(SENSOR_SPECS)

# fault signatures we inject (and label) -- each nudges a subset of sensors
FAULTS = {
    "bearing_wear":    dict(vibration_mm_s=+6.0, acoustic_db=+9.0, temperature_c=+8.0),
    "spindle_overheat":dict(temperature_c=+28.0, spindle_load_pct=+18.0, power_kw=+12.0),
    "tool_wear":       dict(spindle_load_pct=+22.0, power_kw=+9.0, acoustic_db=+7.0),
    "coolant_loss":    dict(temperature_c=+18.0, vibration_mm_s=+3.0),
}
ANOMALY_RATE = 0.06        # fraction of events carrying a fault

# ---------------------------------------------------------------- streaming
# Source is pluggable: "file" (sandbox / CI, fully self-contained) or "kafka"
# (the docker-compose production path). The Spark job reads the same schema either way.
STREAM_SOURCE = os.getenv("STREAM_SOURCE", "file")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "factory.telemetry")
WINDOW = "1 minute"        # tumbling window for the SQL aggregations
WATERMARK = "2 minutes"    # lateness tolerance
ALERT_PROB_THRESHOLD = 0.60  # min model probability to raise an alert

SEED = 42


def spark_session(app="factory-stream-monitor", shuffle=8):
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.appName(app)
         .master(os.getenv("SPARK_MASTER", "local[*]"))
         .config("spark.sql.shuffle.partitions", str(shuffle))
         .config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    return s
