from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent


def prepare_features(data):
    features = data[
        ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    ].copy()
    dates = pd.to_datetime(features.pop("date"), errors="raise")
    features["weight"] = features["weight"].mask(features["weight"] <= 0)
    features["month"] = dates.dt.month
    features["day_of_week"] = dates.dt.dayofweek
    return features


def make_pipeline():
    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            ["distance", "weight", "month", "day_of_week"]
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["pickup", "delivery", "equipment"]
        )
    ])
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(training, evaluation, output_dir, name):
    pipeline = make_pipeline()
    pipeline.fit(prepare_features(training), training["posted_rate"])
    predicted = pipeline.predict(prepare_features(evaluation))
    baseline_rate = (training["posted_rate"] / training["distance"]).median()
    baseline = evaluation["distance"] * baseline_rate
    rows = []
    for label, values in [
        ("Price-per-mile baseline", baseline),
        ("Histogram gradient boosting", predicted)
    ]:
        rows.append({
            "model": label,
            "mae": mean_absolute_error(evaluation["posted_rate"], values),
            "rmse": root_mean_squared_error(evaluation["posted_rate"], values)
        })
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "reports" / f"{name}_results.csv", index=False)
    errors = evaluation[
        ["load_id", "date", "pickup", "delivery", "equipment", "distance", "weight", "posted_rate"]
    ].copy()
    errors["predicted_rate"] = predicted
    errors["absolute_error"] = (errors["posted_rate"] - predicted).abs()
    errors["baseline_absolute_error"] = (errors["posted_rate"] - baseline).abs()
    errors["month"] = errors["date"].dt.strftime("%Y-%m")
    errors.to_csv(output_dir / "reports" / f"{name}_errors.csv", index=False)
    print(f"\n{name} results:", flush=True)
    print(results.to_string(index=False), flush=True)


def export_predictions(pipeline, data_dir, output_dir):
    final = pd.read_csv(data_dir / "validation.csv")
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(data_dir / "december_chart_inputs.csv")
    if not final["load_id"].is_unique or not template["load_id"].is_unique:
        raise ValueError("Prediction input IDs must be unique")
    if len(template) != 12000 or set(template["load_id"]) != set(final["load_id"]):
        raise ValueError("Template IDs must match all 12,000 final inputs")
    lookup = pd.Series(
        pipeline.predict(prepare_features(final)),
        index=final["load_id"]
    )
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].map(lookup)
    december["predicted_rate"] = pipeline.predict(prepare_features(december))
    for values in [submission["predicted_rate"], december["predicted_rate"]]:
        if not np.isfinite(values).all() or not values.gt(0).all():
            raise ValueError("Predicted prices must be finite and positive")
    submission.to_csv(output_dir / "validation_predictions.csv", index=False)
    december.to_csv(output_dir / "december_predictions.csv", index=False)
    subprocess.run([
        sys.executable,
        str(ROOT / "score.py"),
        "--predictions", str(output_dir / "validation_predictions.csv"),
        "--december-predictions", str(output_dir / "december_predictions.csv"),
        "--output-dir", str(output_dir / "scorer_results")
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce the freight rate assessment.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    for name in ["reports", "models", "scorer_results"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    if args.predict_only:
        model_path = args.model_path or output_dir / "models/freight_model.joblib"
        pipeline = joblib.load(model_path)
    else:
        development = pd.read_csv(data_dir / "train_test.csv")
        development["date"] = pd.to_datetime(development["date"], errors="raise")
        training = development[development["date"] < "2025-07-01"].copy()
        validation = development[
            (development["date"] >= "2025-07-01")
            & (development["date"] < "2025-09-01")
        ].copy()
        test = development[development["date"] >= "2025-09-01"].copy()
        if tuple(map(len, [training, validation, test])) != (28806, 9671, 9523):
            raise ValueError("Unexpected development date ranges or row counts")
        print("Training on January-June; evaluating July-August...", flush=True)
        evaluate(training, validation, output_dir, "validation")
        through_august = pd.concat([training, validation], ignore_index=True)
        print("Training through August; evaluating September-October...", flush=True)
        evaluate(through_august, test, output_dir, "local_test")
        print("Fitting final pipeline on all 48,000 labeled loads...", flush=True)
        pipeline = make_pipeline()
        pipeline.fit(prepare_features(development), development["posted_rate"])
        joblib.dump(pipeline, output_dir / "models/freight_model.joblib")
    export_predictions(pipeline, data_dir, output_dir)


if __name__ == "__main__":
    main()
