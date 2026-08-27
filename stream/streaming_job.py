"""The Spark Structured Streaming job -- the heart of the platform.

Reads the telemetry stream (Kafka in production, a file source in the sandbox / CI --
same schema either way), and runs two streaming queries:

  Query 1 (declarative, stateful): a watermarked, tumbling-window Spark-SQL aggregation
    -- per site & machine type, per 1-minute window: throughput and average sensor
    levels. This is the "streaming SQL + watermark" core.

  Query 2 (foreachBatch): applies the persisted MLlib pipeline to each micro-batch,
    raises alerts above a probability threshold, and runs the data-quality/drift check.
    foreachBatch is the standard way to score an ML model inside a streaming query.

Trigger `availableNow` drains all currently-available input in micro-batches and stops
-- perfect for a deterministic sandbox run; production uses a processing-time trigger.
"""
from __future__ import annotations

import json
import shutil

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from stream.config import (ALERT_PROB_THRESHOLD, CHECKPOINT, KAFKA_BOOTSTRAP,
                           KAFKA_TOPIC, MODEL_DIR, STREAM_IN, STREAM_OUT, WATERMARK,
                           WINDOW, spark_session)
from stream.drift import batch_drift, load_reference
from stream.schema import FEATURE_SENSORS, JSON_READ_SCHEMA

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = PipelineModel.load(str(MODEL_DIR))
    return _MODEL


def _read_stream(spark, source: str):
    if source == "kafka":
        raw = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
               .option("subscribe", KAFKA_TOPIC)
               .option("startingOffsets", "earliest").load()
               .selectExpr("CAST(value AS STRING) AS json"))
        parsed = raw.select(F.from_json("json", JSON_READ_SCHEMA).alias("e")).select("e.*")
    else:  # file source
        parsed = (spark.readStream.schema(JSON_READ_SCHEMA)
                  .option("maxFilesPerTrigger", 3)
                  .json(str(STREAM_IN)))
    return parsed.withColumn("event_time", F.to_timestamp("event_time"))


def _foreach_batch(reference):
    def fn(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return
        scored = _model().transform(batch_df)
        scored = scored.withColumn("anomaly_prob", vector_to_array("probability")[1])
        alerts = (scored.filter(F.col("anomaly_prob") >= ALERT_PROB_THRESHOLD)
                  .select("event_id", "event_time", "site", "machine_id", "machine_type",
                          *FEATURE_SENSORS, "anomaly_prob"))
        rows = [r.asDict() for r in alerts.collect()]
        STREAM_OUT.mkdir(parents=True, exist_ok=True)
        with open(STREAM_OUT / "alerts.jsonl", "a") as f:
            for r in rows:
                r["event_time"] = str(r["event_time"])
                f.write(json.dumps(r, default=str) + "\n")

        pdf = batch_df.select(*FEATURE_SENSORS).toPandas()
        drift = batch_drift(pdf, reference)
        drift["batch_id"] = int(batch_id)
        drift["n_events"] = len(pdf)
        drift["n_alerts"] = len(rows)
        with open(STREAM_OUT / "drift.jsonl", "a") as f:
            f.write(json.dumps(drift["_summary"] | {"batch_id": int(batch_id),
                    "n_events": len(pdf), "n_alerts": len(rows)}) + "\n")
    return fn


def run(source="file", await_it=True, clean=True):
    if clean:
        for p in (STREAM_OUT, CHECKPOINT):
            shutil.rmtree(p, ignore_errors=True)
    spark = spark_session("factory-stream-monitor")
    reference = load_reference()
    parsed = _read_stream(spark, source)

    # Query 1 -- watermarked tumbling-window aggregation (throughput + avg sensor levels)
    windowed = (parsed.withWatermark("event_time", WATERMARK)
                .groupBy(F.window("event_time", WINDOW), "site", "machine_type")
                .agg(F.count("*").alias("events"),
                     *[F.round(F.avg(s), 2).alias(f"avg_{s}") for s in FEATURE_SENSORS]))
    q1 = (windowed.selectExpr("window.start AS window_start", "window.end AS window_end",
                              "site", "machine_type", "events",
                              *[f"avg_{s}" for s in FEATURE_SENSORS])
          .writeStream.format("parquet")
          .option("path", str(STREAM_OUT / "window_metrics"))
          .option("checkpointLocation", str(CHECKPOINT / "q1"))
          .outputMode("append").trigger(availableNow=True).start())

    # Query 2 -- model scoring + alerts + drift, per micro-batch
    q2 = (parsed.writeStream.foreachBatch(_foreach_batch(reference))
          .option("checkpointLocation", str(CHECKPOINT / "q2"))
          .trigger(availableNow=True).start())

    if await_it:
        q1.awaitTermination(); q2.awaitTermination()
    return spark, q1, q2


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "file"
    spark, *_ = run(source=src)
    print(f"streaming run complete (source={src}). Outputs in {STREAM_OUT}")
    spark.stop()
