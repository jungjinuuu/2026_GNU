import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import TARGET, add_price_risk_outputs, prediction_suffix
from utils.scoreboard import metric_row, update_score_csv

ROOT = Path(__file__).resolve().parents[2]
TREE_MODEL_DIR = ROOT / "res" / "tree_models"
OPTUNA_DIR = ROOT / "res" / "optuna"
STACKING_DIR = ROOT / "res" / "stacking"
CATBOOST_OPTUNA_DIR = ROOT / "res" / "catboost_optuna"
DNN_DIR = ROOT / "res" / "dnn"
RESULT_DIR = ROOT / "res" / "supervised_ensemble"
DEFAULT_METHODS = ["tree_models", "optuna", "catboost_optuna", "dnn", "stacking"]
TARGET_COL = "target"
PRED_PREFIX = "pred_"
EXCLUDED_PREDICTION_PREFIXES = (
    "pred_ensemble",
    "predicted_",
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ratio_scale_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "ratio_rmse": rmse(y_true, y_pred),
        "ratio_mae": float(mean_absolute_error(y_true, y_pred)),
        "ratio_mean_error": float(np.mean(y_true - y_pred)),
    }


def find_prediction_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if (col.startswith(PRED_PREFIX) or col.endswith("_pred"))
        and not col.startswith(EXCLUDED_PREDICTION_PREFIXES)
        and not col.startswith("residual_")
        and pd.api.types.is_numeric_dtype(df[col])
    )


def filter_prediction_columns(pred_cols: list[str], base_predictions: list[str] | None) -> list[str]:
    if not base_predictions:
        return pred_cols
    requested = set(base_predictions)
    selected = [
        col
        for col in pred_cols
        if col in requested or prediction_suffix(col) in requested
    ]
    missing = sorted(requested - {col for col in selected} - {prediction_suffix(col) for col in selected})
    if missing:
        raise ValueError(
            f"requested prediction columns not found: {missing}. "
            f"Available: {[prediction_suffix(col) for col in pred_cols]} "
            f"(raw columns: {pred_cols})"
        )
    return selected


def canonical_prediction_column(col: str) -> str:
    return f"{PRED_PREFIX}{prediction_suffix(col)}"


def make_run_name(pred_cols: list[str]) -> str:
    return "__".join(prediction_suffix(col) for col in pred_cols)


def add_prediction_columns(
    out: pd.DataFrame,
    source_df: pd.DataFrame,
    source_cols: list[str],
    pred_cols: list[str],
    seen: set[str],
    source_name: str,
) -> None:
    for col in source_cols:
        new_col = canonical_prediction_column(col)
        if new_col in seen:
            new_col = f"{new_col}_{source_name}"
        out[new_col] = source_df[col].to_numpy()
        pred_cols.append(new_col)
        seen.add(new_col)


def load_prediction_files(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    if not paths:
        raise ValueError("at least one prediction file is required")

    base = pd.read_csv(paths[0], low_memory=False)
    if TARGET_COL not in base.columns:
        if TARGET in base.columns:
            base = base.rename(columns={TARGET: TARGET_COL})
        else:
            raise ValueError(f"missing target column in {paths[0]}: {TARGET_COL}")

    source_cols = find_prediction_columns(base)
    out = base.drop(columns=[col for col in source_cols if col.endswith("_pred")], errors="ignore").copy()
    pred_cols: list[str] = []
    seen: set[str] = set()
    add_prediction_columns(out, base, source_cols, pred_cols, seen, paths[0].parent.name)
    for path in paths[1:]:
        df = pd.read_csv(path, low_memory=False)
        if len(df) != len(out):
            raise ValueError(f"row count mismatch: {path} has {len(df)} rows, expected {len(out)}")
        if TARGET_COL not in df.columns:
            if TARGET in df.columns:
                df = df.rename(columns={TARGET: TARGET_COL})
            else:
                raise ValueError(f"missing target column in {path}: {TARGET_COL}")
        if not np.allclose(df[TARGET_COL].to_numpy(), out[TARGET_COL].to_numpy(), equal_nan=True):
            raise ValueError(f"target values do not match first prediction file: {path}")
        add_prediction_columns(out, df, find_prediction_columns(df), pred_cols, seen, path.parent.name)

    return out, pred_cols


def discover_prediction_files(split: str, methods: list[str]) -> list[Path]:
    filename = f"{split}_predictions.csv"
    method_paths = {
        "tree_models": TREE_MODEL_DIR / filename,
        "optuna": OPTUNA_DIR / filename,
        "stacking": STACKING_DIR / filename,
        "catboost_optuna": CATBOOST_OPTUNA_DIR / filename,
        "dnn": DNN_DIR / filename,
    }
    candidates = [method_paths[method] for method in methods]
    return [path for path in candidates if path.exists()]


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0, None)
    total = weights.sum()
    if total == 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def weighted_prediction(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return pred_matrix @ normalize_weights(weights)


def optimize_weights(y_true: np.ndarray, pred_matrix: np.ndarray) -> np.ndarray:
    n_models = pred_matrix.shape[1]
    if n_models == 1:
        return np.array([1.0])

    initial = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(weights: np.ndarray) -> float:
        return rmse(y_true, weighted_prediction(pred_matrix, weights))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        print(f"weight optimization failed, using equal weights: {result.message}")
        return initial
    return normalize_weights(result.x)


def add_ensemble_predictions(df: pd.DataFrame, pred_cols: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    y = out[TARGET_COL].to_numpy()
    pred_matrix = out[pred_cols].to_numpy()

    equal_weights = np.ones(len(pred_cols)) / len(pred_cols)
    best_weights = optimize_weights(y, pred_matrix)

    out["pred_ensemble_equal"] = weighted_prediction(pred_matrix, equal_weights)
    out["residual_ensemble_equal"] = out[TARGET_COL] - out["pred_ensemble_equal"]
    out["pred_ensemble_weighted"] = weighted_prediction(pred_matrix, best_weights)
    out["residual_ensemble_weighted"] = out[TARGET_COL] - out["pred_ensemble_weighted"]
    out = add_price_risk_outputs(out, ["pred_ensemble_equal", "pred_ensemble_weighted"])

    weights = {
        "equal": dict(zip(pred_cols, equal_weights.tolist())),
        "optimized": dict(zip(pred_cols, best_weights.tolist())),
    }
    return out, weights


def build_report(df: pd.DataFrame, pred_cols: list[str], weights: dict[str, dict[str, float]]) -> dict:
    y = df[TARGET_COL].to_numpy()
    report = {
        "input_rows": int(len(df)),
        "prediction_columns": pred_cols,
        "weights": weights,
        "models": {},
    }
    for col in pred_cols + ["pred_ensemble_equal", "pred_ensemble_weighted"]:
        pred = df[col].to_numpy()
        report["models"][col.replace(PRED_PREFIX, "")] = {
            "log_scale": metrics(y, pred),
            "ratio_scale": ratio_scale_metrics(y, pred),
        }
    return report


def ensemble_model_names() -> set[str]:
    return {"ensemble_equal", "ensemble_weighted"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supervised model ensemble from saved validation/test predictions. "
            "By default, it auto-discovers selected method outputs under res/."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help="Supervised method outputs to include when auto-discovering prediction files.",
    )
    parser.add_argument(
        "--base-predictions",
        nargs="+",
        help=(
            "Prediction columns to ensemble after files are loaded. "
            "Examples: lgbm catboost xgboost optuna_lgbm catboost_optuna dnn stacking."
        ),
    )
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument(
        "--prediction-files",
        nargs="+",
        help="Validation prediction CSV files to ensemble. Overrides --predictions.",
    )
    parser.add_argument(
        "--test-prediction-files",
        nargs="+",
        help="Test prediction CSV files matching --prediction-files. Overrides --test-predictions.",
    )
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-weights", action="store_true")
    args = parser.parse_args()

    if args.prediction_files:
        prediction_paths = [Path(path) for path in args.prediction_files]
    elif args.predictions:
        prediction_paths = [Path(args.predictions)]
    else:
        prediction_paths = discover_prediction_files("valid", args.methods)
    if not prediction_paths:
        raise ValueError("no validation prediction files found. Run tree_models.py, optuna.py, catboost_optuna.py, dnn.py, or stacking.py first.")
    df, pred_cols = load_prediction_files(prediction_paths)
    pred_cols = filter_prediction_columns(pred_cols, args.base_predictions)
    run_name = make_run_name(pred_cols)
    output_dir = Path(args.output_dir)
    if output_dir == RESULT_DIR:
        output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pred_cols:
        raise ValueError(f"no prediction columns found with prefix {PRED_PREFIX!r}")
    if len(pred_cols) == 1:
        print(f"only one prediction column found: {pred_cols[0]}. Ensemble will equal that model.")

    ensemble_df, weights = add_ensemble_predictions(df, pred_cols)
    report = {
        "methods": args.methods,
        "run_name": run_name,
        "validation": build_report(ensemble_df, pred_cols, weights),
    }

    if args.test_prediction_files:
        test_prediction_paths = [Path(path) for path in args.test_prediction_files]
    elif args.test_predictions:
        test_prediction_paths = [Path(args.test_predictions)]
    else:
        test_prediction_paths = discover_prediction_files("test", args.methods)
    test_ensemble_df = None
    if all(path.exists() for path in test_prediction_paths):
        test_df, test_pred_cols = load_prediction_files(test_prediction_paths)
        test_pred_cols = [col for col in pred_cols if col in test_df.columns]
        if TARGET_COL not in test_df.columns:
            raise ValueError(f"missing target column in test predictions: {TARGET_COL}")
        if test_pred_cols != pred_cols:
            raise ValueError(f"test prediction columns do not match validation columns: {test_pred_cols} != {pred_cols}")
        test_ensemble_df = test_df.copy()
        pred_matrix = test_ensemble_df[pred_cols].to_numpy()
        equal_weights = np.array(list(weights["equal"].values()))
        optimized_weights = np.array(list(weights["optimized"].values()))
        test_ensemble_df["pred_ensemble_equal"] = weighted_prediction(pred_matrix, equal_weights)
        test_ensemble_df["residual_ensemble_equal"] = test_ensemble_df[TARGET_COL] - test_ensemble_df["pred_ensemble_equal"]
        test_ensemble_df["pred_ensemble_weighted"] = weighted_prediction(pred_matrix, optimized_weights)
        test_ensemble_df["residual_ensemble_weighted"] = (
            test_ensemble_df[TARGET_COL] - test_ensemble_df["pred_ensemble_weighted"]
        )
        test_ensemble_df = add_price_risk_outputs(
            test_ensemble_df,
            ["pred_ensemble_equal", "pred_ensemble_weighted"],
        )
        report["test"] = build_report(test_ensemble_df, pred_cols, weights)

    report_out = output_dir / "metrics.json"

    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    score_rows = []
    for split_name, split_report in report.items():
        if split_name not in {"validation", "test"}:
            continue
        rows = split_report.get("input_rows")
        for model_name, model_report in split_report.get("models", {}).items():
            if model_name not in ensemble_model_names():
                continue
            score_rows.append(
                metric_row(
                    experiment="supervised_ensemble",
                    model=model_name,
                    split=split_name,
                    metrics=model_report["log_scale"],
                    target=TARGET,
                    result_path=report_out.relative_to(ROOT),
                    rows=rows,
                    run_name=run_name,
                    notes=f"base_predictions={','.join(pred_cols)}",
                )
            )
    update_score_csv(Path(args.score_path), score_rows)

    print(f"saved {report_out.relative_to(ROOT)}")
    if args.save_predictions:
        pred_out = output_dir / "valid_predictions_ensemble.csv"
        ensemble_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
        print(f"saved {pred_out.relative_to(ROOT)}")
        if test_ensemble_df is not None:
            test_pred_out = output_dir / "test_predictions_ensemble.csv"
            test_ensemble_df.to_csv(test_pred_out, index=False, encoding="utf-8-sig")
            print(f"saved {test_pred_out.relative_to(ROOT)}")
    if args.save_weights:
        weights_out = output_dir / "weights.json"
        weights_out.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {weights_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
