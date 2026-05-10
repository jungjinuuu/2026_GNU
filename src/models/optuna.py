import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) in sys.path:
    sys.path.remove(str(MODEL_DIR))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import (  # noqa: E402
    DATA_PATH,
    ROOT,
    TARGET,
    load_training_data,
    make_preprocessor,
    make_splits,
    metrics,
    save_predictions,
    select_features,
)
from utils.scoreboard import metric_row, update_score_csv  # noqa: E402


RESULT_DIR = ROOT / "res" / "optuna"


def suggest_lgbm_params(trial) -> dict:
    return {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 500, 2200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 160),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def fit_lgbm_model(x_train, y_train, cat_cols, num_cols, params: dict):
    from lightgbm import LGBMRegressor

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(cat_cols, num_cols)),
            ("model", LGBMRegressor(**params)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna-tuned supervised model.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    import optuna

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(Path(args.data), args.target)
    features, cat_cols, num_cols = select_features(df, args.target)
    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, args.target].to_numpy()
    x_valid = df.loc[valid_mask, features]
    y_valid = df.loc[valid_mask, args.target].to_numpy()
    x_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, args.target].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        model = fit_lgbm_model(x_train, y_train, cat_cols, num_cols, params)
        pred = model.predict(x_valid)
        return metrics(y_valid, pred)["rmse"]

    study = optuna.create_study(direction="minimize", study_name="lgbm_rmse")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    best_params = suggest_lgbm_params(study.best_trial)
    model = fit_lgbm_model(x_train, y_train, cat_cols, num_cols, best_params)
    valid_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)
    valid_metrics = metrics(y_valid, valid_pred)
    test_metrics = metrics(y_test, test_pred)

    report = {
        "target": args.target,
        "model": "optuna_lgbm",
        "split": split_report,
        "features": {
            "count": len(features),
            "categorical": cat_cols,
            "numeric": num_cols,
        },
        "study": {
            "n_trials": len(study.trials),
            "best_value": float(study.best_value),
            "best_params": study.best_params,
        },
        "valid": {"models": {"optuna_lgbm": valid_metrics}},
        "test": {"models": {"optuna_lgbm": test_metrics}},
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    valid_pred_path = output_dir / "valid_predictions.csv"
    test_pred_path = output_dir / "test_predictions.csv"
    save_predictions(df, valid_mask, y_valid, {"optuna_lgbm": valid_pred}, valid_pred_path)
    save_predictions(df, test_mask, y_test, {"optuna_lgbm": test_pred}, test_pred_path)

    score_rows = []
    for split_name, split_metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        score_rows.append(
            metric_row(
                experiment="optuna",
                model="optuna_lgbm",
                split=split_name,
                metrics=split_metrics,
                target=args.target,
                result_path=metrics_path.relative_to(ROOT),
                rows=split_report["rows"][split_name],
                notes=f"n_trials={len(study.trials)}",
            )
        )
    update_score_csv(Path(args.score_path), score_rows)

    print(f"optuna_lgbm valid={valid_metrics} test={test_metrics}")
    print(f"saved {metrics_path.relative_to(ROOT)}")
    print(f"saved {valid_pred_path.relative_to(ROOT)}")
    print(f"saved {test_pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
