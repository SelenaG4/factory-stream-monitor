"""Offline: train the Spark MLlib anomaly-detection Pipeline and register it.

A real MLlib pipeline -- categorical encoding, feature assembly + scaling, a
gradient-boosted-tree classifier -- tuned with cross-validation and scored on a
held-out split, then persisted so the streaming job can load and apply it. The
training feature distribution is saved as the drift baseline; metrics + the model
go to MLflow (with the model registered in the MLflow Model Registry).
"""
from __future__ import annotations

import json

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (OneHotEncoder, StandardScaler, StringIndexer,
                                VectorAssembler)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from stream.config import MODEL_DIR, REFERENCE_JSON, SEED, spark_session
from stream.generate import generate
from stream.schema import CATEGORICAL, FEATURE_SENSORS


def build_pipeline() -> Pipeline:
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                        outputCols=[f"{c}_oh" for c in CATEGORICAL])
    assembler = VectorAssembler(
        inputCols=FEATURE_SENSORS + [f"{c}_oh" for c in CATEGORICAL],
        outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=SEED, maxIter=20, maxBins=32)
    return Pipeline(stages=[*idx, ohe, assembler, scaler, gbt])


def _metrics(pred):
    tp = pred.filter("label = 1 AND prediction = 1").count()
    fp = pred.filter("label = 0 AND prediction = 1").count()
    fn = pred.filter("label = 1 AND prediction = 0").count()
    tn = pred.filter("label = 0 AND prediction = 0").count()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    return dict(precision=precision, recall=recall, f1=f1, accuracy=acc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def save_feature_reference(train_df):
    """Per-sensor training mean/std -> baseline the streaming drift check compares to."""
    aggs = []
    for s in FEATURE_SENSORS:
        aggs += [F.mean(s).alias(f"{s}__mean"), F.stddev(s).alias(f"{s}__std")]
    row = train_df.select(*aggs).collect()[0].asDict()
    ref = {s: {"mean": row[f"{s}__mean"], "std": row[f"{s}__std"]} for s in FEATURE_SENSORS}
    REFERENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_JSON.write_text(json.dumps(ref, indent=2))
    return ref


def main(n_events=30000):
    spark = spark_session("train-anomaly-model")
    pdf = generate(n_events, seed=SEED)
    sdf = spark.createDataFrame(pdf.drop(columns=["fault_type"]))
    train, test = sdf.randomSplit([0.8, 0.2], seed=SEED)
    train.cache(); test.cache()

    grid = (ParamGridBuilder()
            .addGrid(GBTClassifier.maxDepth, [3, 5])
            .build())
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    cv = CrossValidator(estimator=build_pipeline(), estimatorParamMaps=grid,
                        evaluator=evaluator, numFolds=3, seed=SEED, parallelism=2)

    print(f"training on {train.count():,} events, {len(grid)} configs x 3 folds ...")
    model = cv.fit(train)
    best = model.bestModel

    pred = best.transform(test)
    auc = evaluator.evaluate(pred)
    m = _metrics(pred)
    best_depth = best.stages[-1].getMaxDepth()
    print(f"best maxDepth={best_depth}  test AUC={auc:.4f}  "
          f"precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    best.write().overwrite().save(str(MODEL_DIR))
    ref = save_feature_reference(train)
    (MODEL_DIR.parent.parent / "model_metrics.json").write_text(json.dumps({
        "model": "GBTClassifier (Spark MLlib Pipeline)", "auc": round(auc, 4),
        "precision": round(m["precision"], 3), "recall": round(m["recall"], 3),
        "f1": round(m["f1"], 3), "best_max_depth": best_depth, "cv_folds": 3,
        "registry_model": "factory_anomaly_detector", "registry_version": 1,
        "n_train": int(train.count())}, indent=2))
    print(f"saved PipelineModel -> {MODEL_DIR}; feature reference ({len(ref)} sensors) -> {REFERENCE_JSON}")

    _log_mlflow({"auc": auc, **{k: m[k] for k in ("precision", "recall", "f1", "accuracy")},
                 "best_max_depth": best_depth, "n_train": train.count()}, best, spark)
    spark.stop()
    return auc, m


def _log_mlflow(metrics, model, spark):
    try:
        import mlflow, mlflow.spark
    except ImportError:
        print("(mlflow not installed -- skipping tracking/registry)")
        return
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("factory_stream_monitor")
    with mlflow.start_run(run_name="gbt_anomaly_pipeline"):
        mlflow.log_params({"model": "GBTClassifier", "features": "sensors+machine_type+site",
                           "cv_folds": 3})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        try:
            mlflow.spark.log_model(model, "model", registered_model_name="factory_anomaly_detector")
            print("registered model 'factory_anomaly_detector' in the MLflow registry.")
        except Exception as e:
            print(f"(model logged to run; registry step skipped: {e})")
    print("Logged to MLflow (sqlite:///mlflow.db, experiment 'factory_stream_monitor').")


if __name__ == "__main__":
    main()
