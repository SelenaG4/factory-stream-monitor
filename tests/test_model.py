"""Spark-dependent tests: the persisted MLlib pipeline loads and scores sensibly."""
import pytest

from stream.config import MODEL_DIR

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(),
                                reason="model not trained (run: python -m stream.train_model)")


def test_pipeline_loads_and_scores(spark, sample_events):
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    model = PipelineModel.load(str(MODEL_DIR))
    sdf = spark.createDataFrame(sample_events.drop(columns=["fault_type"]))
    scored = (model.transform(sdf)
              .withColumn("anomaly_prob", vector_to_array("probability")[1]))

    rows = scored.select("prediction", "anomaly_prob", "label").collect()
    assert rows, "no rows scored"
    assert all(r["prediction"] in (0.0, 1.0) for r in rows)
    assert all(0.0 <= r["anomaly_prob"] <= 1.0 for r in rows)

    # the model should be clearly better than chance on a fresh sample
    correct = sum(1 for r in rows if int(r["prediction"]) == int(r["label"]))
    assert correct / len(rows) > 0.9
