import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "modeling"
INPUT_PATH = MODEL_DIR / "modeling_dataset.csv"

IN_DOMAIN_TYPES = {"apt", "rh", "officetel", "detached"}
OOD_TYPES = {"urban", "other"}

AREA_MIN_M2 = 5
AREA_DROP_MAX_M2 = 300
AREA_REVIEW_MIN_M2 = 150
ROOM_MAX = 6
SALE_LOW_CONFIDENCE_TXN_COUNT = 10
RATIO_WINSOR_Q_LOW = 0.01
RATIO_WINSOR_Q_HIGH = 0.99


def mode_or_nan(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.mode().iloc[0]


def grouped_fill(series: pd.Series, groups: list[pd.Series], method: str) -> pd.Series:
    filled = series.copy()
    for group in groups:
        if method == "mode":
            mapper = filled.groupby(group, dropna=False).transform(mode_or_nan)
        elif method == "median":
            mapper = filled.groupby(group, dropna=False).transform("median")
        else:
            raise ValueError(f"unknown fill method: {method}")
        filled = filled.fillna(mapper)
    if method == "mode":
        return filled.fillna(mode_or_nan(filled))
    return filled.fillna(filled.median())


def winsorize(series: pd.Series, low_q: float = RATIO_WINSOR_Q_LOW, high_q: float = RATIO_WINSOR_Q_HIGH) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series.copy()
    low = non_null.quantile(low_q)
    high = non_null.quantile(high_q)
    return series.clip(lower=low, upper=high)


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_ood_property_type"] = out["property_type"].isin(OOD_TYPES)
    out["is_in_domain_property_type"] = out["property_type"].isin(IN_DOMAIN_TYPES)
    out["is_exact_duplicate"] = out.duplicated(keep="first")

    out["area_missing_or_zero"] = out["area_m2"].isna() | out["area_m2"].le(0)
    out["area_too_small"] = out["area_m2"].gt(0) & out["area_m2"].lt(AREA_MIN_M2)
    out["area_review_large"] = out["area_m2"].gt(AREA_REVIEW_MIN_M2) & out["area_m2"].le(AREA_DROP_MAX_M2)
    out["area_too_large"] = out["area_m2"].gt(AREA_DROP_MAX_M2)

    out["room_missing"] = out["방갯수"].isna()
    out["room_too_large"] = out["방갯수"].gt(ROOM_MAX)
    out["household_size_missing_or_zero"] = out["세대원수"].isna() | out["세대원수"].le(0)

    out["sale_feature_missing"] = out["sale_txn_count"].isna()
    out["sale_low_confidence"] = out["sale_txn_count"].notna() & out["sale_txn_count"].lt(SALE_LOW_CONFIDENCE_TXN_COUNT)
    out["rent_feature_missing"] = out["rent_txn_count"].isna()

    out["support_ratio_low"] = out["support_ratio"].lt(0.3)
    out["lh_vs_sale_ratio_gt_1"] = out["lh_vs_sale_median_ratio"].gt(1)
    out["jeonse_to_sale_ratio_gt_1"] = out["jeonse_to_sale_ratio_median"].gt(1)
    return out


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["area_m2_clean"] = out["area_m2"].mask(out["area_missing_or_zero"] | out["area_too_small"] | out["area_too_large"])
    out["deposit_per_m2_clean"] = out["deposit_won"] / out["area_m2_clean"]

    out["room_count_clean"] = out["방갯수"].mask(out["room_missing"] | out["room_too_large"])
    room_groups = [
        [out["property_type"], out["주택유형"]],
        [out["property_type"]],
    ]
    out["room_count_imputed"] = grouped_fill(out["room_count_clean"], room_groups, "mode")

    out["household_size_clean"] = out["세대원수"].mask(out["household_size_missing_or_zero"])
    household_groups = [
        [out["property_type"], out["유형"]],
        [out["property_type"]],
    ]
    out["household_size_imputed"] = grouped_fill(out["household_size_clean"], household_groups, "median")

    for col in [
        "lh_vs_rent_median_ratio",
        "lh_vs_sale_median_ratio",
        "jeonse_to_sale_ratio_median",
        "deposit_per_m2_clean",
    ]:
        if col in out.columns:
            clean_col = f"{col}_winsor"
            log_col = f"log1p_{col}_winsor"
            out[clean_col] = winsorize(out[col])
            out[log_col] = np.log1p(out[clean_col])

    return out


def make_preprocessed_datasets(df: pd.DataFrame, drop_sale_missing: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flagged = add_quality_flags(df)
    flagged = add_clean_columns(flagged)

    preprocessed = flagged[
        flagged["is_in_domain_property_type"]
        & ~flagged["is_exact_duplicate"]
        & ~flagged["area_missing_or_zero"]
        & ~flagged["area_too_small"]
        & ~flagged["area_too_large"]
    ].copy()

    if drop_sale_missing:
        preprocessed = preprocessed[~preprocessed["sale_feature_missing"]].copy()

    ood = flagged[
        flagged["is_ood_property_type"]
        & ~flagged["is_exact_duplicate"]
        & ~flagged["area_missing_or_zero"]
        & ~flagged["area_too_small"]
        & ~flagged["area_too_large"]
    ].copy()

    return flagged, preprocessed, ood


def build_report(raw: pd.DataFrame, flagged: pd.DataFrame, preprocessed: pd.DataFrame, ood: pd.DataFrame) -> dict:
    flag_cols = [
        "is_ood_property_type",
        "is_exact_duplicate",
        "area_missing_or_zero",
        "area_too_small",
        "area_review_large",
        "area_too_large",
        "room_missing",
        "room_too_large",
        "household_size_missing_or_zero",
        "sale_feature_missing",
        "sale_low_confidence",
        "rent_feature_missing",
        "support_ratio_low",
        "lh_vs_sale_ratio_gt_1",
        "jeonse_to_sale_ratio_gt_1",
    ]
    report = {
        "input_rows": int(len(raw)),
        "flagged_rows": int(len(flagged)),
        "preprocessed_train_rows": int(len(preprocessed)),
        "preprocessed_ood_rows": int(len(ood)),
        "property_type_counts_input": raw["property_type"].value_counts(dropna=False).to_dict(),
        "property_type_counts_train": preprocessed["property_type"].value_counts(dropna=False).to_dict(),
        "property_type_counts_ood": ood["property_type"].value_counts(dropna=False).to_dict(),
        "flag_counts": {col: int(flagged[col].sum()) for col in flag_cols if col in flagged.columns},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess modeling dataset for analysis/model training.")
    parser.add_argument("--input", default=str(INPUT_PATH), help="input modeling_dataset.csv path")
    parser.add_argument("--output-dir", default=str(MODEL_DIR), help="directory for preprocessed outputs")
    parser.add_argument(
        "--keep-sale-missing",
        action="store_true",
        help="keep rows with missing sale features in the in-domain train output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, low_memory=False)
    flagged, preprocessed, ood = make_preprocessed_datasets(raw, drop_sale_missing=not args.keep_sale_missing)
    report = build_report(raw, flagged, preprocessed, ood)

    flagged_path = output_dir / "modeling_dataset_with_quality_flags.csv"
    train_path = output_dir / "modeling_dataset_preprocessed.csv"
    ood_path = output_dir / "modeling_dataset_ood.csv"
    report_path = output_dir / "preprocessing_report.json"

    flagged.to_csv(flagged_path, index=False, encoding="utf-8-sig")
    preprocessed.to_csv(train_path, index=False, encoding="utf-8-sig")
    ood.to_csv(ood_path, index=False, encoding="utf-8-sig")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input rows={len(raw):,}")
    print(f"saved {train_path.relative_to(ROOT)} rows={len(preprocessed):,}")
    print(f"saved {ood_path.relative_to(ROOT)} rows={len(ood):,}")
    print(f"saved {flagged_path.relative_to(ROOT)} rows={len(flagged):,}")
    print(f"saved {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
