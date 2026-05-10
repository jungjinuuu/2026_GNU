import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import ROOT, TARGET, load_training_data, make_splits
from utils.scoreboard import metric_row, update_score_csv
from supervised_ensemble import optimize_weights, weighted_prediction


TREE_MODELS_DIR = ROOT / "res" / "tree_models"
TREE_ENSEMBLE_DIR = ROOT / "res" / "tree_ensemble"
SUPERVISED_ENSEMBLE_DIR = ROOT / "res" / "supervised_ensemble"
STACKING_DIR = ROOT / "res" / "stacking"
UNSUPERVISED_DATA_PATH = ROOT / "data" / "modeling" / "modeling_dataset_unsupervised.csv"
RESULT_DIR = ROOT / "res" / "unsupervised_ensemble"
TARGET_COL = "target"
UNSUPERVISED_COLUMNS = ["cluster_distance", "robust_z", "isolation_score"]
UNSUPERVISED_RISK_CLIP = 5.0
CONTEXT_COLUMNS = [
    "gu_code",
    "gu_name",
    "ym",
    "contract_date",
    "property_type",
    "주택유형",
    "유형",
    "area_m2_clean",
    "room_count_clean",
    "household_size_clean",
    "deposit_won",
    "rent_deposit_median",
    "rent_txn_count",
    "jeonse_txn_count",
    "sale_price_median",
    "sale_txn_count",
    "is_ood_property_type",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skip missing file: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path, low_memory=False)


def prediction_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if col.startswith("pred_")
        and not col.startswith("pred_ensemble")
        and pd.api.types.is_numeric_dtype(df[col])
    )


def load_tree_predictions(tree_models_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    valid = pd.read_csv(tree_models_dir / "valid_predictions.csv", low_memory=False)
    test = pd.read_csv(tree_models_dir / "test_predictions.csv", low_memory=False)
    pred_cols = prediction_columns(valid)
    if not pred_cols:
        raise ValueError(f"no tree prediction columns found in {tree_models_dir / 'valid_predictions.csv'}")
    missing_test_cols = sorted(set(pred_cols) - set(test.columns))
    if missing_test_cols:
        raise ValueError(f"test predictions missing columns: {missing_test_cols}")
    return valid, test, pred_cols


def load_supervised_ensemble_predictions(
    supervised_ensemble_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    valid_path = supervised_ensemble_dir / "valid_predictions_ensemble.csv"
    test_path = supervised_ensemble_dir / "test_predictions_ensemble.csv"
    if not valid_path.exists() or not test_path.exists():
        return None, None, []

    valid = pd.read_csv(valid_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    if "pred_ensemble_weighted" not in valid.columns or "pred_ensemble_weighted" not in test.columns:
        return None, None, []

    pred_cols = prediction_columns(valid)
    valid["pred_supervised_ensemble"] = valid["pred_ensemble_weighted"]
    test["pred_supervised_ensemble"] = test["pred_ensemble_weighted"]
    return valid, test, pred_cols


def load_tree_ensemble_weights(tree_ensemble_dir: Path, valid_df: pd.DataFrame, pred_cols: list[str]) -> dict:
    metrics_path = tree_ensemble_dir / "metrics.json"
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        weights = report.get("validation", report).get("weights", {}).get("optimized")
        if weights and all(col in weights for col in pred_cols):
            return {col: float(weights[col]) for col in pred_cols}

    print("tree ensemble weights not found; optimizing weights on validation predictions")
    y_valid = valid_df[TARGET_COL].to_numpy()
    pred_matrix = valid_df[pred_cols].to_numpy()
    weights = optimize_weights(y_valid, pred_matrix)
    return dict(zip(pred_cols, weights.astype(float).tolist()))


def add_tree_ensemble_columns(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_cols: list[str],
    weights: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_df.copy()
    test = test_df.copy()
    weight_vector = np.array([weights[col] for col in pred_cols], dtype=float)
    valid["pred_tree_ensemble"] = weighted_prediction(valid[pred_cols].to_numpy(), weight_vector)
    test["pred_tree_ensemble"] = weighted_prediction(test[pred_cols].to_numpy(), weight_vector)
    valid["pred_supervised_ensemble"] = valid["pred_tree_ensemble"]
    test["pred_supervised_ensemble"] = test["pred_tree_ensemble"]
    return valid, test


def add_stacking_predictions(valid_df: pd.DataFrame, test_df: pd.DataFrame, stacking_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    valid_stack = read_csv_if_exists(stacking_dir / "valid_predictions.csv")
    test_stack = read_csv_if_exists(stacking_dir / "test_predictions.csv")
    if valid_stack is None or test_stack is None:
        return valid_df, test_df, []
    if "stacking_pred" not in valid_stack.columns or "stacking_pred" not in test_stack.columns:
        print("skip stacking predictions: stacking_pred column not found")
        return valid_df, test_df, []
    if len(valid_stack) != len(valid_df) or len(test_stack) != len(test_df):
        print("skip stacking predictions: row counts do not match tree prediction files")
        return valid_df, test_df, []

    valid = valid_df.copy()
    test = test_df.copy()
    valid["pred_stacking"] = valid_stack["stacking_pred"].to_numpy()
    test["pred_stacking"] = test_stack["stacking_pred"].to_numpy()
    return valid, test, ["pred_stacking"]


def add_unsupervised_scores(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unsupervised_data_path: Path,
    target: str,
    train_size: float,
    valid_size: float,
    test_size: float,
    random_state: int,
    unsupervised_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_training_data(unsupervised_data_path, target)
    missing_scores = [col for col in unsupervised_columns if col not in df.columns]
    if missing_scores:
        raise ValueError(f"unsupervised score columns not found: {missing_scores}")
    _, valid_mask, test_mask, _ = make_splits(
        df,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )
    extra_cols = [col for col in CONTEXT_COLUMNS + unsupervised_columns if col in df.columns]
    valid_scores = df.loc[valid_mask, extra_cols].reset_index(drop=True)
    test_scores = df.loc[test_mask, extra_cols].reset_index(drop=True)
    if len(valid_scores) != len(valid_df) or len(test_scores) != len(test_df):
        raise ValueError(
            "unsupervised score row counts do not match prediction files. "
            "Use the same split parameters for all model scripts."
        )

    valid = valid_df.reset_index(drop=True).copy()
    test = test_df.reset_index(drop=True).copy()
    for col in extra_cols:
        valid[col] = valid_scores[col].to_numpy()
        test[col] = test_scores[col].to_numpy()
    return valid, test


def base_output_metrics(test_df: pd.DataFrame, output_cols: list[str]) -> dict[str, dict[str, float]]:
    y_test = test_df[TARGET_COL].to_numpy()
    return {
        col: metrics(y_test, test_df[col].to_numpy())
        for col in output_cols
        if col in test_df.columns
    }


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right)
    if pd.isna(value):
        return None
    return float(value)


def risk_score_diagnostics(df: pd.DataFrame, output_cols: list[str]) -> dict[str, dict]:
    diagnostics = {}
    risk = df["unsupervised_risk_score"]
    low_risk = risk <= risk.quantile(0.25)
    high_risk = risk >= risk.quantile(0.75)

    for col in output_cols:
        if col not in df.columns:
            continue
        residual = df[TARGET_COL] - df[col]
        diagnostics[col] = {
            "corr_risk_target": safe_corr(risk, df[TARGET_COL]),
            "corr_risk_residual": safe_corr(risk, residual),
            "low_risk_residual_mean": float(residual[low_risk].mean()),
            "high_risk_residual_mean": float(residual[high_risk].mean()),
            "low_risk_abs_residual_mean": float(residual[low_risk].abs().mean()),
            "high_risk_abs_residual_mean": float(residual[high_risk].abs().mean()),
        }
    return diagnostics


def normalize_unsupervised_scores(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unsupervised_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    valid = valid_df.copy()
    test = test_df.copy()
    score_parts = {}
    score_weights = {col: 1.0 for col in unsupervised_columns}

    for col, weight in score_weights.items():
        median = float(valid[col].median())
        iqr = float(valid[col].quantile(0.75) - valid[col].quantile(0.25))
        scale = iqr if iqr > 1e-9 else float(valid[col].std(ddof=0) + 1e-9)
        valid_z = (valid[col] - median) / scale
        test_z = (test[col] - median) / scale
        valid[f"{col}_risk_z"] = valid_z
        test[f"{col}_risk_z"] = test_z
        score_parts[col] = {
            "weight": weight,
            "median": median,
            "scale": scale,
        }

    risk_cols = [f"{col}_risk_z" for col in score_weights]
    valid["unsupervised_risk_score"] = valid[risk_cols].mul(list(score_weights.values())).sum(axis=1)
    test["unsupervised_risk_score"] = test[risk_cols].mul(list(score_weights.values())).sum(axis=1)

    risk_median = float(valid["unsupervised_risk_score"].median())
    risk_iqr = float(valid["unsupervised_risk_score"].quantile(0.75) - valid["unsupervised_risk_score"].quantile(0.25))
    risk_scale = risk_iqr if risk_iqr > 1e-9 else float(valid["unsupervised_risk_score"].std(ddof=0) + 1e-9)
    valid["unsupervised_risk_score_raw"] = (valid["unsupervised_risk_score"] - risk_median) / risk_scale
    test["unsupervised_risk_score_raw"] = (test["unsupervised_risk_score"] - risk_median) / risk_scale
    valid["unsupervised_risk_score"] = valid["unsupervised_risk_score_raw"].clip(
        -UNSUPERVISED_RISK_CLIP,
        UNSUPERVISED_RISK_CLIP,
    )
    test["unsupervised_risk_score"] = test["unsupervised_risk_score_raw"].clip(
        -UNSUPERVISED_RISK_CLIP,
        UNSUPERVISED_RISK_CLIP,
    )

    report = {
        "score_parts": score_parts,
        "final_score_center": risk_median,
        "final_score_scale": risk_scale,
        "final_score_clip": UNSUPERVISED_RISK_CLIP,
        "valid_raw_score_summary": valid["unsupervised_risk_score_raw"].describe().to_dict(),
        "test_raw_score_summary": test["unsupervised_risk_score_raw"].describe().to_dict(),
        "valid_clipped_score_summary": valid["unsupervised_risk_score"].describe().to_dict(),
        "test_clipped_score_summary": test["unsupervised_risk_score"].describe().to_dict(),
    }
    return valid, test, report


def percentile_from_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = np.sort(reference.dropna().to_numpy())
    if len(ref) == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    ranks = np.searchsorted(ref, values.to_numpy(), side="right") / len(ref)
    return pd.Series(np.clip(ranks, 0.0, 1.0), index=values.index)


def risk_label(score: float) -> str:
    if score >= 80:
        return "high_risk"
    if score >= 60:
        return "caution"
    if score >= 40:
        return "monitor"
    return "low"


def output_suffix(base_col: str) -> str:
    names = {
        "pred_tree_ensemble": "tree_ensemble",
        "pred_supervised_ensemble": "supervised_ensemble",
        "pred_stacking": "stacking",
    }
    return names[base_col]


def top_fraction_capture_rate(y_true: pd.Series, risk_index: pd.Series, fraction: float) -> float:
    n = max(1, int(np.ceil(len(y_true) * fraction)))
    actual_high = set(y_true.nlargest(n).index)
    predicted_high = set(risk_index.nlargest(n).index)
    return len(actual_high & predicted_high) / n


def risk_judgment_metrics(y_true: pd.Series, risk_index: pd.Series) -> dict[str, float]:
    return {
        "risk_spearman": float(y_true.corr(risk_index, method="spearman")),
        "risk_pearson": float(y_true.corr(risk_index, method="pearson")),
        "top10_capture_rate": top_fraction_capture_rate(y_true, risk_index, 0.10),
        "top20_capture_rate": top_fraction_capture_rate(y_true, risk_index, 0.20),
    }


def add_prediction_and_risk_columns(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_cols: list[str],
    tree_pred_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    y_test = test_df[TARGET_COL].to_numpy()
    y_test_series = test_df[TARGET_COL].reset_index(drop=True)
    out = test_df.copy()
    outputs = {}

    if tree_pred_cols:
        valid_disagreement = valid_df[tree_pred_cols].std(axis=1)
        test_disagreement = test_df[tree_pred_cols].std(axis=1)
    else:
        valid_disagreement = pd.Series(np.zeros(len(valid_df)), index=valid_df.index)
        test_disagreement = pd.Series(np.zeros(len(test_df)), index=test_df.index)

    out["model_disagreement_score"] = test_disagreement
    out["model_disagreement_percentile"] = percentile_from_reference(test_disagreement, valid_disagreement)
    out["unsupervised_risk_percentile"] = percentile_from_reference(
        test_df["unsupervised_risk_score"],
        valid_df["unsupervised_risk_score"],
    )

    for base_col in output_cols:
        suffix = output_suffix(base_col)
        valid_ratio = np.expm1(valid_df[base_col])
        test_ratio = np.expm1(test_df[base_col])
        ratio_percentile = percentile_from_reference(pd.Series(test_ratio), pd.Series(valid_ratio))
        out[f"predicted_ratio_{suffix}"] = test_ratio
        out[f"predicted_ratio_percentile_{suffix}"] = ratio_percentile

        variants = {
            suffix: {
                "uses_unsupervised_features": False,
                "risk_index_formula": (
                    "100 * (0.85 * predicted_ratio_percentile "
                    "+ 0.15 * model_disagreement_percentile)"
                ),
                "risk_index": 100
                * (
                    0.85 * out[f"predicted_ratio_percentile_{suffix}"]
                    + 0.15 * out["model_disagreement_percentile"]
                ),
            },
            f"unsupervised_{suffix}": {
                "uses_unsupervised_features": True,
                "risk_index_formula": (
                    "100 * (0.60 * predicted_ratio_percentile "
                    "+ 0.30 * unsupervised_risk_percentile "
                    "+ 0.10 * model_disagreement_percentile)"
                ),
                "risk_index": 100
                * (
                    0.60 * out[f"predicted_ratio_percentile_{suffix}"]
                    + 0.30 * out["unsupervised_risk_percentile"]
                    + 0.10 * out["model_disagreement_percentile"]
                ),
            },
        }

        for output_name, spec in variants.items():
            fair_deposit_col = f"predicted_fair_deposit_won_{output_name}"
            risk_index_col = f"risk_judgment_index_{output_name}"
            risk_label_col = f"risk_label_{output_name}"
            out[f"predicted_ratio_{output_name}"] = test_ratio
            out[f"predicted_ratio_percentile_{output_name}"] = ratio_percentile
            if "rent_deposit_median" in out.columns:
                out[fair_deposit_col] = out["rent_deposit_median"] * out[f"predicted_ratio_{output_name}"]
                if "deposit_won" in out.columns:
                    out[f"deposit_gap_won_{output_name}"] = out["deposit_won"] - out[fair_deposit_col]
                    out[f"deposit_gap_ratio_{output_name}"] = out["deposit_won"] / out[fair_deposit_col].replace(0, np.nan)
            out[risk_index_col] = spec["risk_index"]
            out[risk_label_col] = out[risk_index_col].map(risk_label)
            outputs[output_name] = {
                "base_column": base_col,
                "uses_unsupervised_features": spec["uses_unsupervised_features"],
                "prediction_target_metrics": metrics(y_test, out[base_col].to_numpy()),
                "risk_judgment_metrics": risk_judgment_metrics(y_test_series, out[risk_index_col].reset_index(drop=True)),
                "price_column": fair_deposit_col if "rent_deposit_median" in out.columns else None,
                "risk_index_column": risk_index_col,
                "risk_label_column": risk_label_col,
                "risk_index_formula": spec["risk_index_formula"],
            }

    return {"outputs": outputs}, out


def choose_output_columns(
    attach_unsupervised_to: str,
    stacking_cols: list[str],
) -> list[str]:
    if attach_unsupervised_to in {"supervised_ensemble", "tree_ensemble"}:
        return ["pred_supervised_ensemble"]
    elif attach_unsupervised_to == "stacking":
        if not stacking_cols:
            raise ValueError(
                "attach_unsupervised_to='stacking' requires stacking predictions. "
                "Run `python src/models/stacking.py --save-predictions` first."
            )
        return stacking_cols
    elif attach_unsupervised_to == "both":
        if not stacking_cols:
            print("stacking predictions not found; falling back to tree ensemble + unsupervised scores")
        return ["pred_supervised_ensemble"] + stacking_cols
    else:
        raise ValueError(f"unknown attach_unsupervised_to: {attach_unsupervised_to}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fair deposit predictions and separate risk judgment indices.")
    parser.add_argument("--tree-models-dir", default=str(TREE_MODELS_DIR))
    parser.add_argument("--tree-ensemble-dir", default=str(TREE_ENSEMBLE_DIR))
    parser.add_argument("--supervised-ensemble-dir", default=str(SUPERVISED_ENSEMBLE_DIR))
    parser.add_argument("--stacking-dir", default=str(STACKING_DIR))
    parser.add_argument("--unsupervised-data", default=str(UNSUPERVISED_DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument(
        "--unsupervised-score-columns",
        nargs="+",
        default=UNSUPERVISED_COLUMNS,
        help="Score columns from modeling_dataset_unsupervised.csv used in unsupervised risk index.",
    )
    parser.add_argument(
        "--attach-unsupervised-to",
        choices=["supervised_ensemble", "tree_ensemble", "stacking", "both"],
        default="both",
        help=(
            "Which supervised output is used for fair deposit and risk judgment outputs: "
            "supervised_ensemble, stacking, or both. tree_ensemble is kept as an alias."
        ),
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print detailed base metrics and risk diagnostics.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid, test, tree_pred_cols = load_supervised_ensemble_predictions(Path(args.supervised_ensemble_dir))
    weights = None
    if valid is None or test is None:
        valid, test, tree_pred_cols = load_tree_predictions(Path(args.tree_models_dir))
        weights = load_tree_ensemble_weights(Path(args.tree_ensemble_dir), valid, tree_pred_cols)
        valid, test = add_tree_ensemble_columns(valid, test, tree_pred_cols, weights)
    valid, test, stacking_cols = add_stacking_predictions(valid, test, Path(args.stacking_dir))
    valid, test = add_unsupervised_scores(
        valid,
        test,
        Path(args.unsupervised_data),
        args.target,
        args.train_size,
        args.valid_size,
        args.test_size,
        args.random_state,
        args.unsupervised_score_columns,
    )

    valid, test, unsupervised_report = normalize_unsupervised_scores(valid, test, args.unsupervised_score_columns)
    output_cols = choose_output_columns(
        args.attach_unsupervised_to,
        stacking_cols,
    )
    report, test_predictions = add_prediction_and_risk_columns(valid, test, output_cols, tree_pred_cols)
    base_output_cols = tree_pred_cols + ["pred_supervised_ensemble"] + stacking_cols
    report.update(
        {
            "target": args.target,
            "attach_unsupervised_to": args.attach_unsupervised_to,
            "method": "separate_price_prediction_and_risk_judgment",
            "rows": {
                "valid": int(len(valid)),
                "test": int(len(test)),
            },
            "base_output_metrics": base_output_metrics(test, base_output_cols),
            "risk_score_diagnostics": {
                "valid": risk_score_diagnostics(valid, output_cols),
                "test": risk_score_diagnostics(test, output_cols),
            },
            "feature_groups": {
                "tree_base_predictions": tree_pred_cols,
                "supervised_ensemble_output": ["pred_supervised_ensemble"],
                "stacking_output": stacking_cols,
                "unsupervised_scores": args.unsupervised_score_columns,
            },
            "selected_prediction_outputs": output_cols,
            "tree_prediction_columns": tree_pred_cols,
            "supervised_ensemble_weights": weights,
            "stacking_columns": stacking_cols,
            "unsupervised_columns": args.unsupervised_score_columns,
            "unsupervised_score": unsupervised_report,
        }
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    score_rows = [
        metric_row(
            experiment="ensemble",
            model=name,
            split="test",
            metrics=output_report["prediction_target_metrics"],
            target=args.target,
            result_path=metrics_path.relative_to(ROOT),
            rows=report["rows"]["test"],
            notes=(
                f"uses_unsupervised_features={output_report['uses_unsupervised_features']}; "
                f"risk_spearman={output_report['risk_judgment_metrics']['risk_spearman']:.6f}; "
                f"top20_capture_rate={output_report['risk_judgment_metrics']['top20_capture_rate']:.6f}"
            ),
        )
        for name, output_report in report["outputs"].items()
    ]
    update_score_csv(Path(args.score_path), score_rows)
    for name, output_report in report["outputs"].items():
        print(
            f"{name} "
            f"prediction={output_report['prediction_target_metrics']} "
            f"risk={output_report['risk_judgment_metrics']}"
        )
    if args.verbose:
        for name, base_metrics in report["base_output_metrics"].items():
            print(f"base {name}", base_metrics)
        for split_name, diagnostics in report["risk_score_diagnostics"].items():
            for base_col, values in diagnostics.items():
                print(
                    f"diagnostic {split_name} {base_col}",
                    {
                        "corr_risk_residual": values["corr_risk_residual"],
                        "low_risk_abs_residual_mean": values["low_risk_abs_residual_mean"],
                        "high_risk_abs_residual_mean": values["high_risk_abs_residual_mean"],
                    },
                )

    if args.save_predictions:
        pred_path = output_dir / "test_predictions.csv"
        test_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"saved {pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
