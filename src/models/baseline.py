import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import (  # noqa: E402
    BASELINE_RESULT_DIR,
    DATA_PATH,
    ROOT,
    SCORE_PATH,
    TARGET,
    load_training_data,
    make_splits,
    metrics,
    save_predictions,
)
from utils.scoreboard import metric_row, update_score_csv  # noqa: E402


BASELINE_MODELS = ["global_median", "gu_property_median"]


def predict_group_median(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    group_cols: list[str],
    fallback: float,
) -> np.ndarray:
    medians = train.groupby(group_cols, dropna=False)[target].median().rename("_group_median")
    pred = valid.merge(medians, left_on=group_cols, right_index=True, how="left")["_group_median"]
    return pred.fillna(fallback).to_numpy()


def run_simple_baselines(
    df: pd.DataFrame,
    target: str,
    train_size: float,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict, tuple[pd.Series, pd.Series, pd.Series]]:
    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )
    train = df.loc[train_mask].copy()
    valid = df.loc[valid_mask].copy()
    test = df.loc[test_mask].copy()
    y_valid = valid[target].to_numpy()
    y_test = test[target].to_numpy()

    global_median = float(train[target].median())
    valid_predictions = {
        "global_median": np.full(len(valid), global_median),
        "gu_property_median": predict_group_median(train, valid, target, ["gu_code", "property_type"], global_median),
    }
    test_predictions = {
        "global_median": np.full(len(test), global_median),
        "gu_property_median": predict_group_median(train, test, target, ["gu_code", "property_type"], global_median),
    }
    scores = {
        "target": target,
        "split": split_report,
        "valid": {
            "models": {name: metrics(y_valid, pred) for name, pred in valid_predictions.items()},
        },
        "test": {
            "models": {name: metrics(y_test, pred) for name, pred in test_predictions.items()},
        },
    }
    return {"valid": valid_predictions, "test": test_predictions}, scores, (train_mask, valid_mask, test_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple non-ML baselines for LH market-relative target.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=str(BASELINE_RESULT_DIR))
    parser.add_argument("--score-path", default=str(SCORE_PATH))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(Path(args.data), args.target)
    predictions, scores, masks = run_simple_baselines(
        df,
        args.target,
        args.train_size,
        args.valid_size,
        args.test_size,
        args.random_state,
    )
    _, valid_mask, test_mask = masks
    y_valid = df.loc[valid_mask, args.target].to_numpy()
    y_test = df.loc[test_mask, args.target].to_numpy()

    score_path = output_dir / "metrics.json"
    score_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    score_rows = []
    rows = scores["split"]["rows"].get("test")
    for model_name in BASELINE_MODELS:
        score_rows.append(
            metric_row(
                experiment="baseline",
                model=model_name,
                split="test",
                metrics=scores["test"]["models"][model_name],
                target=args.target,
                result_path=score_path.relative_to(ROOT),
                rows=rows,
            )
        )
    update_score_csv(Path(args.score_path), score_rows)
    print(f"saved {score_path.relative_to(ROOT)}")
    if args.save_predictions:
        valid_pred_path = output_dir / "valid_predictions.csv"
        test_pred_path = output_dir / "test_predictions.csv"
        save_predictions(df, valid_mask, y_valid, predictions["valid"], valid_pred_path)
        save_predictions(df, test_mask, y_test, predictions["test"], test_pred_path)
        print(f"saved {valid_pred_path.relative_to(ROOT)}")
        print(f"saved {test_pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
