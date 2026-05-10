from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = [
    "updated_at",
    "experiment",
    "run_name",
    "model",
    "split",
    "rmse",
    "mae",
    "r2",
    "target",
    "rows",
    "result_path",
    "notes",
]
SCORE_KEY_COLUMNS = ["experiment", "run_name", "model", "split", "target"]


def metric_row(
    experiment: str,
    model: str,
    split: str,
    metrics: dict,
    target: str,
    result_path: Path,
    rows: int | None = None,
    run_name: str = "default",
    notes: str = "",
) -> dict:
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "run_name": run_name,
        "model": model,
        "split": split,
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "target": target,
        "rows": rows,
        "result_path": str(result_path),
        "notes": notes,
    }


def update_score_csv(score_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    score_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if "split" in new_df.columns:
        new_df = new_df[new_df["split"].eq("test")].copy()
    if new_df.empty:
        return
    for col in SCORE_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[SCORE_COLUMNS]

    if score_path.exists():
        old_df = pd.read_csv(score_path, low_memory=False)
        if "run_name" not in old_df.columns:
            old_df["run_name"] = "legacy"
        for col in SCORE_COLUMNS:
            if col not in old_df.columns:
                old_df[col] = None
        old_df = old_df[SCORE_COLUMNS]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=SCORE_KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["split", "rmse", "experiment", "model"], na_position="last")
    combined.to_csv(score_path, index=False, encoding="utf-8-sig")
