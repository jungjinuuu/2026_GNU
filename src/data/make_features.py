from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = ROOT / "data" / "clean"
FEATURE_DIR = ROOT / "data" / "features"


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def make_rent_features() -> pd.DataFrame:
    rent = safe_read(CLEAN_DIR / "rent_clean.csv")
    if rent.empty:
        return pd.DataFrame(
            columns=[
                "gu_code",
                "ym",
                "property_type",
                "rent_txn_count",
                "jeonse_txn_count",
                "rent_deposit_median",
                "rent_deposit_per_m2_median",
                "monthly_rent_median",
            ]
        )

    grouped = (
        rent.groupby(["gu_code", "gu_name", "ym", "property_type"], dropna=False)
        .agg(
            rent_txn_count=("deposit_won", "size"),
            jeonse_txn_count=("rent_type", lambda x: (x == "jeonse").sum()),
            rent_deposit_median=("deposit_won", "median"),
            rent_deposit_mean=("deposit_won", "mean"),
            rent_deposit_per_m2_median=("deposit_per_m2", "median"),
            rent_deposit_per_m2_mean=("deposit_per_m2", "mean"),
            monthly_rent_median=("monthly_rent_won", "median"),
            area_m2_median=("area_m2", "median"),
        )
        .reset_index()
    )
    return grouped


def make_sale_features() -> pd.DataFrame:
    sale = safe_read(CLEAN_DIR / "sale_clean.csv")
    if sale.empty:
        return pd.DataFrame(
            columns=[
                "gu_code",
                "ym",
                "property_type",
                "sale_txn_count",
                "sale_price_median",
                "sale_price_per_m2_median",
            ]
        )

    grouped = (
        sale.groupby(["gu_code", "gu_name", "ym", "property_type"], dropna=False)
        .agg(
            sale_txn_count=("sale_price_won", "size"),
            sale_price_median=("sale_price_won", "median"),
            sale_price_mean=("sale_price_won", "mean"),
            sale_price_per_m2_median=("sale_price_per_m2", "median"),
            sale_price_per_m2_mean=("sale_price_per_m2", "mean"),
            sale_area_m2_median=("area_m2", "median"),
        )
        .reset_index()
    )
    return grouped


def make_index_features() -> pd.DataFrame:
    candidates = sorted((ROOT / "data" / "raw").glob("*부동산원*.*")) + sorted((ROOT / "data" / "raw").glob("*index*.*"))
    if not candidates:
        return pd.DataFrame(columns=["gu_code", "ym"])

    frames = []
    for path in candidates:
        try:
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(path, encoding="cp949"))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(FEATURE_DIR / "index_features_raw_unprocessed.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(columns=["gu_code", "ym"])


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    rent_features = make_rent_features()
    sale_features = make_sale_features()
    index_features = make_index_features()

    rent_features.to_csv(FEATURE_DIR / "rent_features.csv", index=False, encoding="utf-8-sig")
    sale_features.to_csv(FEATURE_DIR / "sale_features.csv", index=False, encoding="utf-8-sig")
    index_features.to_csv(FEATURE_DIR / "index_features.csv", index=False, encoding="utf-8-sig")

    print(f"rent_features rows={len(rent_features):,}")
    print(f"sale_features rows={len(sale_features):,}")
    print(f"index_features rows={len(index_features):,}")


if __name__ == "__main__":
    main()
