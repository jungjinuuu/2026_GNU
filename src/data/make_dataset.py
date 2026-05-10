from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "clean"
FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "data" / "modeling"

START_YM = "2022-01"
END_YM = "2025-12"


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    lh = safe_read(CLEAN_DIR / "lh_clean.csv")
    if lh.empty:
        raise FileNotFoundError("data/clean/lh_clean.csv가 없습니다. src/data/clean_data.py를 먼저 실행하세요.")

    lh = lh[(lh["ym"] >= START_YM) & (lh["ym"] <= END_YM)].copy()
    rent_features = safe_read(FEATURE_DIR / "rent_features.csv")
    sale_features = safe_read(FEATURE_DIR / "sale_features.csv")
    index_features = safe_read(FEATURE_DIR / "index_features.csv")

    dataset = lh.copy()
    merge_keys = ["gu_code", "ym", "property_type"]

    if not rent_features.empty:
        dataset = dataset.merge(
            rent_features.drop(columns=["gu_name"], errors="ignore"),
            on=merge_keys,
            how="left",
        )

    if not sale_features.empty:
        dataset = dataset.merge(
            sale_features.drop(columns=["gu_name"], errors="ignore"),
            on=merge_keys,
            how="left",
        )

    if not index_features.empty:
        index_keys = [key for key in ["gu_code", "ym"] if key in index_features.columns]
        if index_keys:
            dataset = dataset.merge(index_features, on=index_keys, how="left")

    if "rent_deposit_median" in dataset.columns:
        dataset["lh_vs_rent_median_ratio"] = dataset["deposit_won"] / dataset["rent_deposit_median"]
    if "sale_price_median" in dataset.columns:
        dataset["lh_vs_sale_median_ratio"] = dataset["deposit_won"] / dataset["sale_price_median"]
    if {"rent_deposit_median", "sale_price_median"}.issubset(dataset.columns):
        dataset["jeonse_to_sale_ratio_median"] = dataset["rent_deposit_median"] / dataset["sale_price_median"]

    out_path = MODEL_DIR / "modeling_dataset.csv"
    dataset.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_path.relative_to(ROOT)} rows={len(dataset):,} cols={len(dataset.columns)}")


if __name__ == "__main__":
    main()
