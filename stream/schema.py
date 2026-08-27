"""The telemetry event schema, defined once and shared by the producer, the
streaming job, and the tests so the wire format never drifts between them."""
from __future__ import annotations

from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField,
                               StructType, TimestampType)

from stream.config import SENSORS

# event: identity + timestamp + the sensor readings (+ a label, only for the
# synthetic generator / evaluation; the streaming job never reads the label)
EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_time", TimestampType()),
        StructField("site", StringType()),
        StructField("machine_id", StringType()),
        StructField("machine_type", StringType()),
    ]
    + [StructField(s, DoubleType()) for s in SENSORS]
    + [
        StructField("label", IntegerType()),       # 1 = anomaly (synthetic ground truth)
        StructField("fault_type", StringType()),    # None when healthy
    ]
)

# the columns the model consumes as raw features (engineered inside the Pipeline)
FEATURE_SENSORS = list(SENSORS)
CATEGORICAL = ["machine_type", "site"]

# For reading events off the wire (JSON files / Kafka value): event_time arrives as an
# ISO-8601 string, so we read it as a string and cast with to_timestamp in the job.
JSON_READ_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("event_time", StringType()),
        StructField("site", StringType()),
        StructField("machine_id", StringType()),
        StructField("machine_type", StringType()),
    ]
    + [StructField(s, DoubleType()) for s in SENSORS]
    + [StructField("label", IntegerType()), StructField("fault_type", StringType())]
)
