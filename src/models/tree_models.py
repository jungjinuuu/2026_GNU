import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import (
    DATA_PATH,
    EXCLUDE_COLUMNS,
    ROOT,
    TARGET,
    TREE_RESULT_DIR,
    load_training_data,
    make_preprocessor,
    make_splits,
    metrics,
    save_predictions,
    select_features,
)
from utils.scoreboard import metric_row, update_score_csv


def fit_lgbm(x_train, y_train, x_valid, y_valid, cat_cols, num_cols):
    from lightgbm import LGBMRegressor

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(cat_cols, num_cols)),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=1200,
                    learning_rate=0.03,
                    num_leaves=63,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    model.fit(y=y_train, X=x_train)
    return model, model.predict(x_valid)


def fit_xgboost(x_train, y_train, x_valid, y_valid, cat_cols, num_cols):
    from xgboost import XGBRegressor

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(cat_cols, num_cols)),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=900,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    model.fit(y=y_train, X=x_train)
    return model, model.predict(x_valid)


def fit_catboost(x_train, y_train, x_valid, y_valid, cat_cols, num_cols):
    from catboost import CatBoostRegressor, Pool

    train = x_train.copy()
    valid = x_valid.copy()
    for col in cat_cols:
        train[col] = train[col].astype("string").fillna("__MISSING__")
        valid[col] = valid[col].astype("string").fillna("__MISSING__")
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        valid[col] = pd.to_numeric(valid[col], errors="coerce")

    cat_idx = [train.columns.get_loc(col) for col in cat_cols]
    train_pool = Pool(train, y_train, cat_features=cat_idx)
    valid_pool = Pool(valid, y_valid, cat_features=cat_idx)
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=100)
    return model, model.predict(valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree model baselines for LH market-relative target.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(TREE_RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--save-features", action="store_true")
    args = parser.parse_args()

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

    model_fns = {
        "lgbm": fit_lgbm,
        "catboost": fit_catboost,
        "xgboost": fit_xgboost,
    }
    valid_predictions = {}
    test_predictions = {}
    scores = {
        "target": args.target,
        "split": split_report,
        "features": {
            "count": len(features),
            "categorical": cat_cols,
            "numeric": num_cols,
            "excluded": sorted(EXCLUDE_COLUMNS),
        },
        "valid": {"models": {}},
        "test": {"models": {}},
    }

    for name, fit_fn in model_fns.items():
        try:
            model, valid_pred = fit_fn(x_train, y_train, x_valid, y_valid, cat_cols, num_cols)
            if hasattr(model, "predict"):
                if name == "catboost":
                    test_input = x_test.copy()
                    for col in cat_cols:
                        test_input[col] = test_input[col].astype("string").fillna("__MISSING__")
                    for col in num_cols:
                        test_input[col] = pd.to_numeric(test_input[col], errors="coerce")
                    test_pred = model.predict(test_input)
                else:
                    test_pred = model.predict(x_test)
            else:
                raise RuntimeError(f"{name} did not return a predict-capable model")
        except Exception as exc:
            scores["valid"]["models"][name] = {"error": repr(exc)}
            scores["test"]["models"][name] = {"error": repr(exc)}
            print(f"skip {name}: {exc}")
            continue
        valid_predictions[name] = valid_pred
        test_predictions[name] = test_pred
        scores["valid"]["models"][name] = metrics(y_valid, valid_pred)
        scores["test"]["models"][name] = metrics(y_test, test_pred)

    if len(valid_predictions) >= 2:
        valid_ensemble_pred = np.mean(np.column_stack(list(valid_predictions.values())), axis=1)
        test_ensemble_pred = np.mean(np.column_stack(list(test_predictions.values())), axis=1)
        valid_predictions["ensemble_mean"] = valid_ensemble_pred
        test_predictions["ensemble_mean"] = test_ensemble_pred
        scores["valid"]["models"]["ensemble_mean"] = metrics(y_valid, valid_ensemble_pred)
        scores["test"]["models"]["ensemble_mean"] = metrics(y_test, test_ensemble_pred)

    score_path = output_dir / "metrics.json"
    valid_pred_path = output_dir / "valid_predictions.csv"
    test_pred_path = output_dir / "test_predictions.csv"
    score_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(df, valid_mask, y_valid, valid_predictions, valid_pred_path)
    save_predictions(df, test_mask, y_test, test_predictions, test_pred_path)
    score_rows = []
    for split_name in ["valid", "test"]:
        rows = scores["split"]["rows"].get(split_name)
        for model_name, model_metrics in scores[split_name]["models"].items():
            if "error" in model_metrics:
                continue
            score_rows.append(
                metric_row(
                    experiment="tree_models",
                    model=model_name,
                    split=split_name,
                    metrics=model_metrics,
                    target=args.target,
                    result_path=score_path.relative_to(ROOT),
                    rows=rows,
                )
            )
    update_score_csv(Path(args.score_path), score_rows)
    print(f"saved {score_path.relative_to(ROOT)}")
    print(f"saved {valid_pred_path.relative_to(ROOT)}")
    print(f"saved {test_pred_path.relative_to(ROOT)}")
    if args.save_features:
        feature_path = output_dir / "features.json"
        feature_path.write_text(
            json.dumps(
                {"features": features, "categorical": cat_cols, "numeric": num_cols},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {feature_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
