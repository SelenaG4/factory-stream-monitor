"""Emit telemetry events into the stream.

Two interchangeable sinks behind one interface:
  * "file"  -- write micro-batches of JSON lines into STREAM_IN (Spark file-source
               reads each new file as it lands). Fully self-contained; used in the
               sandbox and CI.
  * "kafka" -- publish to a Kafka topic (the docker-compose production path).

Both emit the identical JSON event shape, so the Spark job is source-agnostic.
"""
from __future__ import annotations

import json
import time

from stream.config import (KAFKA_BOOTSTRAP, KAFKA_TOPIC, STREAM_IN, STREAM_SOURCE)
from stream.generate import generate


def _event_json(row) -> str:
    d = row._asdict() if hasattr(row, "_asdict") else dict(row)
    d["event_time"] = d["event_time"].isoformat()
    return json.dumps(d, default=str)


def to_files(df, batch_size=500, out_dir=STREAM_IN):
    """Write the events as JSON-lines files, batch_size events per file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        path = out_dir / f"events_{start:08d}.json"
        path.write_text("\n".join(_event_json(r) for r in chunk.itertuples(index=False)))
        n += 1
    return n


def to_kafka(df, bootstrap=KAFKA_BOOTSTRAP, topic=KAFKA_TOPIC, rate_per_sec=200):
    """Publish events to Kafka at a controlled rate (the live production path)."""
    from kafka import KafkaProducer  # lazy: only needed on the kafka path

    producer = KafkaProducer(bootstrap_servers=bootstrap,
                             value_serializer=lambda v: v.encode("utf-8"))
    delay = 1.0 / rate_per_sec
    for r in df.itertuples(index=False):
        producer.send(topic, _event_json(r))
        time.sleep(delay)
    producer.flush()


def main(n_events=8000, source=STREAM_SOURCE):
    df = generate(n_events)
    if source == "kafka":
        to_kafka(df)
        print(f"published {len(df)} events to Kafka topic '{KAFKA_TOPIC}'")
    else:
        files = to_files(df)
        print(f"wrote {files} JSON batch files ({len(df)} events) to {STREAM_IN}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
