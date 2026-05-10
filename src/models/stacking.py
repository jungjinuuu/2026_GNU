import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import (
    DATA_PATH,
    OUTPUT_CONTEXT_COLUMNS,
    ROOT,
    TARGET,
    add_price_risk_outputs,
    load_training_data,
    make_splits,
    metrics,
    select_features,
)
from utils.scoreboard import metric_row, update_score_csv
from tree_models import fit_catboost, fit_lgbm, fit_xgboost
from supervised_ensemble import (
    discover_prediction_files,
    filter_prediction_columns,
    load_prediction_files,
)


RESULT_DIR = ROOT / "res" / "stacking"
DEFAULT_METHODS = ["tree_models", "optuna"]
DEFAULT_BASE_MODELS = ["lgbm", "catboost", "xgboost"]


def month_folds(train_df: pd.DataFrame, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    months = np.array(sorted(train_df["ym"].dropna().unique()))
    if len(months) < n_folds:
        raise ValueError(f"not enough months for {n_folds} folds: {len(months)} months")

    month_chunks = np.array_split(months, n_folds)
    folds = []
    for valid_months in month_chunks:
        valid_mask = train_df["ym"].isin(valid_months).to_numpy()
        train_mask = ~valid_mask
        folds.append((np.where(train_mask)[0], np.where(valid_mask)[0]))
    return folds


def fit_base_model(model_name, fit_fn, x_train, y_train, x_valid, y_valid, cat_cols, num_cols):
    try:
        _, pred = fit_fn(x_train, y_train, x_valid, y_valid, cat_cols, num_cols)
    except Exception as exc:
        return None, repr(exc)
    return pred, None


def build_oof_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    target: str,
    n_folds: int,
    model_fns: dict,
) -> tuple[pd.DataFrame, dict]:
    folds = month_folds(train_df, n_folds)
    oof = pd.DataFrame(index=train_df.index)
    oof[target] = train_df[target]
    fold_reports = []
    model_errors = {name: [] for name in model_fns}

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_train = train_df.iloc[tr_idx]
        fold_valid = train_df.iloc[va_idx]
        x_tr = fold_train[features]
        y_tr = fold_train[target].to_numpy()
        x_va = fold_valid[features]
        y_va = fold_valid[target].to_numpy()

        fold_report = {
            "fold": fold_idx,
            "valid_months": sorted(fold_valid["ym"].unique().tolist()),
            "train_rows": int(len(fold_train)),
            "valid_rows": int(len(fold_valid)),
            "models": {},
        }
        for model_name, fit_fn in model_fns.items():
            pred, error = fit_base_model(model_name, fit_fn, x_tr, y_tr, x_va, y_va, cat_cols, num_cols)
            if error:
                model_errors[model_name].append(error)
                fold_report["models"][model_name] = {"error": error}
                continue
            oof.loc[fold_valid.index, f"{model_name}_oof"] = pred
            fold_report["models"][model_name] = metrics(y_va, pred)

        fold_reports.append(fold_report)

    available_models = [
        name
        for name in model_fns
        if f"{name}_oof" in oof.columns and oof[f"{name}_oof"].notna().all()
    ]
    if not available_models:
        raise RuntimeError("no complete OOF predictions were created")

    report = {
        "folds": fold_reports,
        "available_models": available_models,
        "model_errors": {k: v for k, v in model_errors.items() if v},
    }
    return oof, report


def fit_holdout_base_predictions(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    target: str,
    model_fns: dict,
    available_models: list[str],
) -> tuple[pd.DataFrame, dict]:
    context_cols = [col for col in OUTPUT_CONTEXT_COLUMNS if col in valid_df.columns]
    holdout = valid_df[context_cols].copy()
    holdout[target] = valid_df[target]
    reports = {}

    x_train = train_df[features]
    y_train = train_df[target].to_numpy()
    x_valid = valid_df[features]
    y_valid = valid_df[target].to_numpy()

    for model_name in available_models:
        pred, error = fit_base_model(
            model_name,
            model_fns[model_name],
            x_train,
            y_train,
            x_valid,
            y_valid,
            cat_cols,
            num_cols,
        )
        if error:
            reports[model_name] = {"error": error}
            continue
        holdout[f"{model_name}_pred"] = pred
        reports[model_name] = metrics(y_valid, pred)

    complete_models = [
        name
        for name in available_models
        if f"{name}_pred" in holdout.columns and holdout[f"{name}_pred"].notna().all()
    ]
    if not complete_models:
        raise RuntimeError("no holdout base predictions were created")
    return holdout, {"models": reports, "available_models": complete_models}


def fit_meta_model(oof: pd.DataFrame, holdout: pd.DataFrame, target: str, available_models: list[str]) -> tuple[np.ndarray, dict]:
    oof_cols = [f"{name}_oof" for name in available_models]
    holdout_cols = [f"{name}_pred" for name in available_models]
    meta_train = oof[oof_cols].to_numpy()
    meta_target = oof[target].to_numpy()
    meta_valid = holdout[holdout_cols].to_numpy()

    meta_model = RidgeCV(alphas=np.logspace(-4, 4, 25), fit_intercept=True)
    meta_model.fit(meta_train, meta_target)
    pred = meta_model.predict(meta_valid)
    report = {
        "model": "RidgeCV",
        "alpha": float(meta_model.alpha_),
        "intercept": float(meta_model.intercept_),
        "coef": dict(zip(available_models, meta_model.coef_.astype(float).tolist())),
    }
    return pred, report


def fit_saved_prediction_stacking(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    x_valid = valid_df[pred_cols].to_numpy()
    y_valid = valid_df["target"].to_numpy()
    x_test = test_df[pred_cols].to_numpy()
    y_test = test_df["target"].to_numpy()

    meta_model = RidgeCV(alphas=np.logspace(-4, 4, 25), fit_intercept=True)
    meta_model.fit(x_valid, y_valid)

    out = test_df.copy()
    out["stacking_pred"] = meta_model.predict(x_test)
    out["stacking_residual"] = out["target"] - out["stacking_pred"]
    out = add_price_risk_outputs(out, pred_cols + ["stacking_pred"])

    report = {
        "target": TARGET,
        "mode": "saved_prediction_stacking",
        "base_prediction_columns": pred_cols,
        "meta_model": {
            "model": "RidgeCV",
            "alpha": float(meta_model.alpha_),
            "intercept": float(meta_model.intercept_),
            "coef": dict(zip(pred_cols, meta_model.coef_.astype(float).tolist())),
        },
        "valid_stacking": metrics(y_valid, meta_model.predict(x_valid)),
        "test_stacking": metrics(y_test, out["stacking_pred"].to_numpy()),
        "split": {
            "rows": {
                "valid": int(len(valid_df)),
                "test": int(len(test_df)),
            }
        },
    }
    return out, report


def save_outputs(
    output_dir: Path,
    report: dict,
    oof: pd.DataFrame,
    valid_holdout: pd.DataFrame,
    test_holdout: pd.DataFrame,
    save_predictions: bool,
    score_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    score_rows = [
        metric_row(
            experiment="stacking",
            model="stacking",
            split="valid",
            metrics=report["valid_stacking"],
            target=report["target"],
            result_path=metrics_path.relative_to(ROOT),
            rows=report["split"]["rows"]["valid"],
        ),
        metric_row(
            experiment="stacking",
            model="stacking",
            split="test",
            metrics=report["test_stacking"],
            target=report["target"],
            result_path=metrics_path.relative_to(ROOT),
            rows=report["split"]["rows"]["test"],
        ),
    ]
    update_score_csv(score_path, score_rows)
    print(f"saved {metrics_path.relative_to(ROOT)}")
    if save_predictions:
        oof_path = output_dir / "oof_predictions.csv"
        valid_holdout_path = output_dir / "valid_predictions.csv"
        test_holdout_path = output_dir / "test_predictions.csv"
        oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
        valid_holdout.to_csv(valid_holdout_path, index=False, encoding="utf-8-sig")
        test_holdout.to_csv(test_holdout_path, index=False, encoding="utf-8-sig")
        print(f"saved {oof_path.relative_to(ROOT)}")
        print(f"saved {valid_holdout_path.relative_to(ROOT)}")
        print(f"saved {test_holdout_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF stacking for tree model predictions.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--base-models",
        nargs="+",
        choices=DEFAULT_BASE_MODELS,
        default=DEFAULT_BASE_MODELS,
        help="Internal OOF base models to train when prediction files are not provided.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help="Saved supervised method outputs to stack when using prediction files or auto-discovery.",
    )
    parser.add_argument("--prediction-files", nargs="+", help="Validation prediction CSV files for saved-prediction stacking.")
    parser.add_argument("--test-prediction-files", nargs="+", help="Test prediction CSV files for saved-prediction stacking.")
    parser.add_argument(
        "--base-predictions",
        nargs="+",
        help="Prediction columns to stack. Examples: lgbm optuna_lgbm catboost.",
    )
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if args.prediction_files or args.test_prediction_files:
        valid_paths = [Path(path) for path in (args.prediction_files or discover_prediction_files("valid", args.methods))]
        test_paths = [Path(path) for path in (args.test_prediction_files or discover_prediction_files("test", args.methods))]
        valid_df, pred_cols = load_prediction_files(valid_paths)
        test_df, test_pred_cols = load_prediction_files(test_paths)
        pred_cols = filter_prediction_columns(pred_cols, args.base_predictions)
        test_pred_cols = [col for col in pred_cols if col in test_df.columns]
        if test_pred_cols != pred_cols:
            raise ValueError(f"test prediction columns do not match validation columns: {test_pred_cols} != {pred_cols}")
        test_out, report = fit_saved_prediction_stacking(valid_df, test_df, pred_cols)

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        update_score_csv(
            Path(args.score_path),
            [
                metric_row(
                    experiment="stacking",
                    model="stacking",
                    split="test",
                    metrics=report["test_stacking"],
                    target=report["target"],
                    result_path=metrics_path.relative_to(ROOT),
                    rows=report["split"]["rows"]["test"],
                    notes=f"mode=saved_prediction_stacking; base_predictions={','.join(pred_cols)}",
                )
            ],
        )
        print(f"saved {metrics_path.relative_to(ROOT)}")
        if args.save_predictions:
            valid_out = valid_df.copy()
            valid_out["stacking_pred"] = RidgeCV(alphas=np.logspace(-4, 4, 25), fit_intercept=True).fit(
                valid_df[pred_cols].to_numpy(),
                valid_df["target"].to_numpy(),
            ).predict(valid_df[pred_cols].to_numpy())
            valid_out["stacking_residual"] = valid_out["target"] - valid_out["stacking_pred"]
            valid_out = add_price_risk_outputs(valid_out, pred_cols + ["stacking_pred"])
            valid_path = output_dir / "valid_predictions.csv"
            test_path = output_dir / "test_predictions.csv"
            valid_out.to_csv(valid_path, index=False, encoding="utf-8-sig")
            test_out.to_csv(test_path, index=False, encoding="utf-8-sig")
            print(f"saved {valid_path.relative_to(ROOT)}")
            print(f"saved {test_path.relative_to(ROOT)}")
        return

    df = load_training_data(Path(args.data), args.target)
    features, cat_cols, num_cols = select_features(df, args.target)
    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    test_df = df.loc[test_mask].copy()

    all_model_fns = {
        "lgbm": fit_lgbm,
        "catboost": fit_catboost,
        "xgboost": fit_xgboost,
    }
    model_fns = {name: all_model_fns[name] for name in args.base_models}

    oof, oof_report = build_oof_predictions(
        train_df=train_df,
        features=features,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target=args.target,
        n_folds=args.n_folds,
        model_fns=model_fns,
    )
    valid_holdout, valid_holdout_report = fit_holdout_base_predictions(
        train_df=train_df,
        valid_df=valid_df,
        features=features,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target=args.target,
        model_fns=model_fns,
        available_models=oof_report["available_models"],
    )
    test_holdout, test_holdout_report = fit_holdout_base_predictions(
        train_df=train_df,
        valid_df=test_df,
        features=features,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target=args.target,
        model_fns=model_fns,
        available_models=oof_report["available_models"],
    )

    available_models = valid_holdout_report["available_models"]
    valid_stack_pred, meta_report = fit_meta_model(oof, valid_holdout, args.target, available_models)
    test_stack_pred, _ = fit_meta_model(oof, test_holdout, args.target, available_models)
    valid_holdout["stacking_pred"] = valid_stack_pred
    valid_holdout["stacking_residual"] = valid_holdout[args.target] - valid_holdout["stacking_pred"]
    valid_pred_cols = [f"{name}_pred" for name in available_models] + ["stacking_pred"]
    valid_holdout = add_price_risk_outputs(valid_holdout, valid_pred_cols)
    test_holdout["stacking_pred"] = test_stack_pred
    test_holdout["stacking_residual"] = test_holdout[args.target] - test_holdout["stacking_pred"]
    test_pred_cols = [f"{name}_pred" for name in available_models] + ["stacking_pred"]
    test_holdout = add_price_risk_outputs(test_holdout, test_pred_cols)

    y_valid = valid_holdout[args.target].to_numpy()
    y_test = test_holdout[args.target].to_numpy()
    report = {
        "target": args.target,
        "split": split_report | {"n_folds": args.n_folds},
        "features": {
            "count": len(features),
            "categorical": cat_cols,
            "numeric": num_cols,
        },
        "oof": oof_report,
        "valid_base_models": valid_holdout_report["models"],
        "test_base_models": test_holdout_report["models"],
        "meta_model": meta_report,
        "valid_stacking": metrics(y_valid, valid_stack_pred),
        "test_stacking": metrics(y_test, test_stack_pred),
    }
    save_outputs(
        output_dir,
        report,
        oof,
        valid_holdout,
        test_holdout,
        args.save_predictions,
        Path(args.score_path),
    )


if __name__ == "__main__":
    main()
