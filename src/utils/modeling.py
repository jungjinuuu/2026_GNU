from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "modeling" / "modeling_dataset_preprocessed.csv"
BASELINE_RESULT_DIR = ROOT / "res" / "baseline"
TREE_RESULT_DIR = ROOT / "res" / "tree_models"
SCORE_PATH = ROOT / "res" / "score.csv"
TARGET = "log1p_lh_vs_rent_median_ratio_winsor"

EXCLUDE_COLUMNS = {
    TARGET,
    "lh_vs_rent_median_ratio",
    "lh_vs_rent_median_ratio_winsor",
    "lh_vs_sale_median_ratio",
    "lh_vs_sale_median_ratio_winsor",
    "log1p_lh_vs_sale_median_ratio_winsor",
    "jeonse_to_sale_ratio_median",
    "jeonse_to_sale_ratio_median_winsor",
    "log1p_jeonse_to_sale_ratio_median_winsor",
    "deposit_per_m2",
    "deposit_per_m2_clean",
    "deposit_per_m2_clean_winsor",
    "log1p_deposit_per_m2_clean_winsor",
    "deposit_won",
    "support_won",
    "support_ratio",
    "support_ratio_low",
    "tenant_deposit_won",
    "monthly_rent_won",
    "lh_vs_sale_ratio_gt_1",
    "jeonse_to_sale_ratio_gt_1",
    "contract_date",
    "주소",
}

ID_COLUMNS = [
    "gu_code",
    "gu_name",
    "ym",
    "property_type",
    "주택유형",
    "유형",
]

OUTPUT_CONTEXT_COLUMNS = [
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

CATEGORICAL_COLUMNS = {
    "gu_code",
    "gu_name",
    "ym",
    "property_type",
    "주택유형",
    "유형",
    "cluster_id",
}

DEFAULT_GROUP_COLS = ["gu_code", "ym", "property_type"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    group_cols: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size + valid_size + test_size must equal 1.0")
    group_cols = group_cols or DEFAULT_GROUP_COLS
    missing = [col for col in group_cols + ["ym"] if col not in df.columns]
    if missing:
        raise ValueError(f"missing split columns: {missing}")

    rng = np.random.default_rng(random_state)
    groups = df[group_cols].astype("string").copy()
    groups["_year"] = df["ym"].astype("string").str[:4]
    unique_groups = groups.drop_duplicates().reset_index(drop=True)

    split_parts = []
    year_summary = {}
    for year, year_groups in unique_groups.groupby("_year", dropna=False):
        indices = year_groups.index.to_numpy().copy()
        rng.shuffle(indices)
        n_groups = len(indices)
        n_train = int(np.floor(n_groups * train_size))
        n_valid = int(np.floor(n_groups * valid_size))
        n_test = n_groups - n_train - n_valid
        if n_groups >= 3:
            if n_train == 0:
                n_train = 1
            if n_valid == 0:
                n_valid = 1
            n_test = n_groups - n_train - n_valid
            if n_test == 0:
                n_test = 1
                n_train = max(1, n_train - 1)

        labels = np.array(["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test)
        assigned = year_groups.loc[indices].copy()
        assigned["_split"] = labels[: len(assigned)]
        split_parts.append(assigned)
        year_summary[str(year)] = {
            "groups": int(n_groups),
            "train_groups": int((assigned["_split"] == "train").sum()),
            "valid_groups": int((assigned["_split"] == "valid").sum()),
            "test_groups": int((assigned["_split"] == "test").sum()),
        }

    split_map = pd.concat(split_parts, ignore_index=True)
    row_splits = groups.merge(split_map, on=group_cols + ["_year"], how="left")["_split"]
    if row_splits.isna().any():
        raise RuntimeError("failed to assign every row to a split")

    train_mask = pd.Series(row_splits.eq("train").to_numpy(), index=df.index)
    valid_mask = pd.Series(row_splits.eq("valid").to_numpy(), index=df.index)
    test_mask = pd.Series(row_splits.eq("test").to_numpy(), index=df.index)
    report = {
        "strategy": "year_stratified_group_split",
        "group_cols": group_cols,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "random_state": random_state,
        "rows": {
            "train": int(train_mask.sum()),
            "valid": int(valid_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "years": year_summary,
    }
    return train_mask, valid_mask, test_mask, report


def make_split(df: pd.DataFrame, valid_start_ym: str | None = None) -> tuple[pd.Series, pd.Series]:
    train_mask, valid_mask, _, _ = make_splits(df)
    return train_mask, valid_mask


def select_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    usable = [col for col in df.columns if col not in EXCLUDE_COLUMNS and col != target]
    cat_cols = [
        col
        for col in usable
        if col in CATEGORICAL_COLUMNS or df[col].dtype == "object" or str(df[col].dtype).startswith("string")
    ]
    num_cols = [col for col in usable if col not in cat_cols and is_numeric_dtype(df[col])]
    features = cat_cols + num_cols
    dropped = sorted(set(usable) - set(features))
    if dropped:
        print(f"drop non-numeric/non-categorical columns: {dropped}")
    return features, cat_cols, num_cols


def make_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def load_training_data(data_path: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if target not in df.columns:
        raise ValueError(f"target column not found: {target}")
    return df[df[target].notna()].copy()


def prediction_suffix(pred_col: str) -> str:
    if pred_col.startswith("pred_"):
        return pred_col.removeprefix("pred_")
    if pred_col.endswith("_pred"):
        return pred_col.removesuffix("_pred")
    return pred_col


def risk_label(score: float) -> str:
    if score >= 80:
        return "high_risk"
    if score >= 60:
        return "caution"
    if score >= 40:
        return "monitor"
    return "low"


def add_price_risk_outputs(out: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    out = out.copy()
    for pred_col in pred_cols:
        if pred_col not in out.columns:
            continue
        suffix = prediction_suffix(pred_col)
        ratio_col = f"predicted_ratio_{suffix}"
        fair_deposit_col = f"predicted_fair_deposit_won_{suffix}"
        risk_index_col = f"risk_judgment_index_{suffix}"
        risk_label_col = f"risk_label_{suffix}"

        out[ratio_col] = np.expm1(out[pred_col])
        out[f"predicted_ratio_percentile_{suffix}"] = out[ratio_col].rank(pct=True, method="average").fillna(0.0)
        if "rent_deposit_median" in out.columns:
            out[fair_deposit_col] = out["rent_deposit_median"] * out[ratio_col]
            if "deposit_won" in out.columns:
                out[f"deposit_gap_won_{suffix}"] = out["deposit_won"] - out[fair_deposit_col]
                out[f"deposit_gap_ratio_{suffix}"] = out["deposit_won"] / out[fair_deposit_col].replace(0, np.nan)
        out[risk_index_col] = 100 * out[f"predicted_ratio_percentile_{suffix}"]
        out[risk_label_col] = out[risk_index_col].map(risk_label)
    return out


def build_prediction_output(
    df: pd.DataFrame,
    mask: pd.Series,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    out_cols = [col for col in OUTPUT_CONTEXT_COLUMNS if col in df.columns]
    out = df.loc[mask, out_cols].copy()
    out["target"] = y_true
    pred_cols = []
    for name, pred in predictions.items():
        pred_col = f"pred_{name}"
        out[pred_col] = pred
        out[f"residual_{name}"] = out["target"] - out[pred_col]
        pred_cols.append(pred_col)
    return add_price_risk_outputs(out, pred_cols)


def save_predictions(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    y_valid: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    out = build_prediction_output(df, valid_mask, y_valid, predictions)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
