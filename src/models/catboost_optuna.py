import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) in sys.path:
    sys.path.remove(str(MODEL_DIR))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    EXCLUDE_COLUMNS,
    ROOT,
    TARGET,
    load_training_data,
    make_splits,
    metrics,
    save_predictions,
    select_features,
)
from utils.scoreboard import metric_row, update_score_csv  # noqa: E402


RESULT_DIR = ROOT / "res" / "catboost_optuna"
MODEL_NAME = "catboost_optuna"
MISSING_VALUE = "__MISSING__"


def add_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    combo_specs = {
        "gu_property": ["gu_name", "property_type"],
        "gu_ym": ["gu_name", "ym"],
        "property_ym": ["property_type", "ym"],
        "housing_type_ym": ["주택유형", "ym"],
        "detail_type_ym": ["유형", "ym"],
    }
    for new_col, cols in combo_specs.items():
        if all(col in out.columns for col in cols):
            values = [out[col].astype("string").fillna(MISSING_VALUE) for col in cols]
            out[new_col] = values[0]
            for value in values[1:]:
                out[new_col] = out[new_col] + "__" + value
    return out


def prepare_catboost_frame(
    df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
) -> pd.DataFrame:
    out = df[features].copy()
    for col in cat_cols:
        out[col] = out[col].astype("string").fillna(MISSING_VALUE)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def suggest_catboost_params(trial, loss_function: str, random_state: int) -> dict:
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params = {
        "loss_function": loss_function,
        "eval_metric": "RMSE",
        "iterations": trial.suggest_int("iterations", 800, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 64, 254),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 80),
        "bootstrap_type": bootstrap_type,
        "random_seed": random_state,
        "allow_writing_files": False,
        "verbose": False,
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.55, 1.0)
    return params


def fit_catboost_model(x_train, y_train, x_valid, y_valid, cat_cols: list[str], params: dict):
    from catboost import CatBoostRegressor, Pool

    cat_idx = [x_train.columns.get_loc(col) for col in cat_cols]
    train_pool = Pool(x_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(x_valid, y_valid, cat_features=cat_idx)
    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=150,
        verbose=False,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna-tuned CatBoost with native categorical features.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--loss-function",
        default="RMSE",
        help="CatBoost loss function. Examples: RMSE, MAE, Huber:delta=1.0, Quantile:alpha=0.5.",
    )
    parser.add_argument("--no-combo-features", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    import optuna

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(Path(args.data), args.target)
    if not args.no_combo_features:
        df = add_combo_features(df)

    features, cat_cols, num_cols = select_features(df, args.target)
    combo_cols = [
        col
        for col in ["gu_property", "gu_ym", "property_ym", "housing_type_ym", "detail_type_ym"]
        if col in df.columns and col not in cat_cols
    ]
    cat_cols = cat_cols + combo_cols
    features = [col for col in features if col not in combo_cols] + combo_cols
    num_cols = [col for col in features if col not in cat_cols and pd.api.types.is_numeric_dtype(df[col])]

    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_train = prepare_catboost_frame(df.loc[train_mask], features, cat_cols, num_cols)
    y_train = df.loc[train_mask, args.target].to_numpy()
    x_valid = prepare_catboost_frame(df.loc[valid_mask], features, cat_cols, num_cols)
    y_valid = df.loc[valid_mask, args.target].to_numpy()
    x_test = prepare_catboost_frame(df.loc[test_mask], features, cat_cols, num_cols)
    y_test = df.loc[test_mask, args.target].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_catboost_params(trial, args.loss_function, args.random_state)
        model = fit_catboost_model(x_train, y_train, x_valid, y_valid, cat_cols, params)
        pred = model.predict(x_valid)
        return metrics(y_valid, pred)["rmse"]

    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    study = optuna.create_study(direction="minimize", study_name="catboost_rmse", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    best_params = suggest_catboost_params(study.best_trial, args.loss_function, args.random_state)
    model = fit_catboost_model(x_train, y_train, x_valid, y_valid, cat_cols, best_params)
    valid_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)
    valid_metrics = metrics(y_valid, valid_pred)
    test_metrics = metrics(y_test, test_pred)

    report = {
        "target": args.target,
        "model": MODEL_NAME,
        "split": split_report,
        "features": {
            "count": len(features),
            "categorical": cat_cols,
            "numeric": num_cols,
            "excluded": sorted(EXCLUDE_COLUMNS),
            "combo_features": not args.no_combo_features,
        },
        "study": {
            "n_trials": len(study.trials),
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "loss_function": args.loss_function,
        },
        "valid": {"models": {MODEL_NAME: valid_metrics}},
        "test": {"models": {MODEL_NAME: test_metrics}},
    }

    metrics_path = output_dir / "metrics.json"
    valid_pred_path = output_dir / "valid_predictions.csv"
    test_pred_path = output_dir / "test_predictions.csv"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(df, valid_mask, y_valid, {MODEL_NAME: valid_pred}, valid_pred_path)
    save_predictions(df, test_mask, y_test, {MODEL_NAME: test_pred}, test_pred_path)

    score_rows = []
    run_name = f"{MODEL_NAME}_{args.loss_function.replace(':', '_').replace('=', '-')}_{len(study.trials)}trials"
    for split_name, split_metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        score_rows.append(
            metric_row(
                experiment="catboost_optuna",
                run_name=run_name,
                model=MODEL_NAME,
                split=split_name,
                metrics=split_metrics,
                target=args.target,
                result_path=metrics_path.relative_to(ROOT),
                rows=split_report["rows"][split_name],
                notes=f"n_trials={len(study.trials)},loss={args.loss_function}",
            )
        )
    update_score_csv(Path(args.score_path), score_rows)

    print(f"{MODEL_NAME} valid={valid_metrics} test={test_metrics}")
    print(f"saved {metrics_path.relative_to(ROOT)}")
    print(f"saved {valid_pred_path.relative_to(ROOT)}")
    print(f"saved {test_pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
