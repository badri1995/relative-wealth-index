"""Portable CSV interface to the selected-feature RWI regression experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def fit_and_evaluate(train, test, features, id_column="village_id", target="RWI", group_column=None):
    if not isinstance(features, list) or not features or not all(isinstance(name, str) and name for name in features):
        raise ValueError("Features must be a nonempty JSON list of column names.")
    if len(features) != len(set(features)):
        raise ValueError("Feature names must be unique.")
    reserved = {id_column, target}
    if group_column:
        reserved.add(group_column)
    if reserved.intersection(features):
        raise ValueError("Identifiers, target and split group must not be predictors.")
    required = reserved | set(features)
    for name, frame in (("train", train), ("test", test)):
        if frame.empty or not frame.columns.is_unique or not required.issubset(frame.columns):
            raise ValueError(f"{name} must be nonempty with unique columns including {sorted(required)}.")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} contains missing required values.")
        identifiers = frame[id_column].astype(str).str.strip()
        if identifiers.eq("").any() or identifiers.duplicated().any():
            raise ValueError(f"{name} identifiers must be nonempty and unique.")
        values = frame[features + [target]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} predictors and targets must be finite numbers.")
    if set(train[id_column].astype(str).str.strip()) & set(test[id_column].astype(str).str.strip()):
        raise ValueError("A village appears in both training and test data.")
    if group_column:
        train_groups = train[group_column].astype(str).str.strip()
        test_groups = test[group_column].astype(str).str.strip()
        if train_groups.eq("").any() or test_groups.eq("").any() or set(train_groups) & set(test_groups):
            raise ValueError("Spatial groups must be nonempty and disjoint across splits.")
    model = RandomForestRegressor(
        max_features="sqrt", max_samples=0.7, min_samples_split=75,
        n_estimators=80, n_jobs=-1, oob_score=True, random_state=0,
    )
    model.fit(train[features].astype(float), train[target].astype(float))
    predictions = model.predict(test[features].astype(float))
    residuals = predictions - test[target].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    target_range = float(train[target].astype(float).max() - train[target].astype(float).min())
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "features": features, "mae": float(np.abs(residuals).mean()), "rmse": rmse,
        "rmse_over_training_range": rmse / target_range if target_range > 1e-12 else None,
        "group_column": group_column,
        "note": "Supplied split evaluation; spatial independence requires a meaningful group design.",
    }
    return model, predictions, metrics


def synthetic_data():
    generator = np.random.default_rng(42)
    table = pd.DataFrame({
        "village_id": [f"synthetic_{index:04d}" for index in range(300)],
        "satellite_proxy": generator.uniform(0, 1, 300),
        "nightlight_proxy": generator.uniform(0, 1, 300),
    })
    table["RWI"] = 0.7 * table["satellite_proxy"] + 0.3 * table["nightlight_proxy"]
    return table.iloc[:240].copy(), table.iloc[240:].copy(), ["satellite_proxy", "nightlight_proxy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--id-column", default="village_id")
    parser.add_argument("--target", default="RWI")
    parser.add_argument("--group-column")
    parser.add_argument("--output", type=Path, default=Path("outputs/regression"))
    args = parser.parse_args()
    if args.demo:
        if any([args.train, args.test, args.features, args.group_column]) or args.id_column != "village_id" or args.target != "RWI":
            parser.error("--demo cannot be combined with data or schema options")
        train, test, features = synthetic_data()
    else:
        if not all([args.train, args.test, args.features]):
            parser.error("supply --train, --test and --features, or use --demo")
        identifier_types = {args.id_column: str}
        if args.group_column:
            identifier_types[args.group_column] = str
        train = pd.read_csv(args.train, dtype=identifier_types)
        test = pd.read_csv(args.test, dtype=identifier_types)
        features = json.loads(args.features.read_text())
    _, predictions, metrics = fit_and_evaluate(train, test, features, args.id_column, args.target, args.group_column)
    metrics["mode"] = "synthetic demonstration, not research results" if args.demo else "user supplied data"
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test[args.id_column], "target": test[args.target], "prediction": predictions}).to_csv(args.output / "predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
