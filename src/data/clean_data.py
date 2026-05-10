from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RTMS_DIR = RAW_DIR / "rtms_api"
CLEAN_DIR = ROOT / "data" / "clean"

LAWD_PATH = RAW_DIR / "국토교통부_전국 법정동_20250807.csv"
LH_PATH = RAW_DIR / "한국토지주택공사_임대주택공급현황_전세임대_20251120.csv"

START_YM = "2022-01"
END_YM = "2025-12"


def to_number(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    if not text:
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def pick(df: pd.DataFrame, names: list[str], default: object = np.nan) -> pd.Series:
    existing = [name for name in names if name in df.columns]
    if not existing:
        return pd.Series(default, index=df.index)
    result = df[existing[0]]
    for name in existing[1:]:
        result = result.combine_first(df[name])
    return result


def make_gu_map() -> pd.DataFrame:
    lawd = pd.read_csv(LAWD_PATH, encoding="utf-8-sig")
    seoul = lawd[(lawd["시도명"] == "서울특별시") & lawd["삭제일자"].isna()].copy()
    gu = seoul[seoul["시군구명"].notna()].copy()
    gu["gu_code"] = gu["법정동코드"].astype(str).str[:5]
    gu["gu_name"] = gu["시군구명"]
    return gu[["gu_code", "gu_name"]].drop_duplicates().sort_values("gu_code")


def normalize_lh_housing_type(value: object) -> object:
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == "아파트":
        return "apt"
    if value in {"연립주택", "다세대주택"}:
        return "rh"
    if value == "오피스텔":
        return "officetel"
    if value in {"다가구용단독주택", "단독주택", "다중주택"}:
        return "detached"
    if value == "도시형생활주택":
        return "urban"
    return "other"


def clean_lh(gu_map: pd.DataFrame) -> pd.DataFrame:
    lh = pd.read_csv(LH_PATH, encoding="cp949")
    lh["gu_name"] = lh["주소"].str.extract(r"서울특별시\s+([가-힣]+구)", expand=False)
    lh = lh[lh["gu_name"].notna()].copy()
    lh = lh.merge(gu_map, on="gu_name", how="left")

    lh["contract_date"] = pd.to_datetime(lh["계약일자"], errors="coerce")
    lh["ym"] = lh["contract_date"].dt.to_period("M").astype("string")
    lh["property_type"] = lh["주택유형"].map(normalize_lh_housing_type)
    lh["area_m2"] = pd.to_numeric(lh["전용면적"], errors="coerce")
    lh["deposit_won"] = pd.to_numeric(lh["전세금"], errors="coerce")
    lh["support_won"] = pd.to_numeric(lh["전세지원금"], errors="coerce")
    lh["monthly_rent_won"] = pd.to_numeric(lh["월임대료"], errors="coerce")
    lh["tenant_deposit_won"] = lh["deposit_won"] - lh["support_won"]
    lh["deposit_per_m2"] = lh["deposit_won"] / lh["area_m2"].replace(0, np.nan)
    lh["support_ratio"] = lh["support_won"] / lh["deposit_won"].replace(0, np.nan)

    keep = [
        "gu_code",
        "gu_name",
        "ym",
        "contract_date",
        "property_type",
        "주택유형",
        "유형",
        "방갯수",
        "area_m2",
        "세대원수",
        "deposit_won",
        "support_won",
        "tenant_deposit_won",
        "monthly_rent_won",
        "deposit_per_m2",
        "support_ratio",
        "주소",
    ]
    return lh[keep]


def read_rtms_files(trade_type: str) -> pd.DataFrame:
    files = sorted(RTMS_DIR.glob(f"{trade_type}_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path, dtype=str) for path in files], ignore_index=True)


def clean_rent(gu_map: pd.DataFrame) -> pd.DataFrame:
    raw = read_rtms_files("rent")
    if raw.empty:
        return raw

    out = pd.DataFrame(index=raw.index)
    out["trade_type"] = "rent"
    out["property_type"] = pick(raw, ["property_type"])
    out["gu_code"] = pick(raw, ["sggCd", "request_gu_code"]).astype("string").str[:5]
    out["dong_name"] = pick(raw, ["umdNm", "umdNm"])
    out["building_name"] = pick(raw, ["aptNm", "mhouseNm", "offiNm", "단지"])
    out["house_type"] = pick(raw, ["houseType"])
    out["jibun"] = pick(raw, ["jibun"])
    out["contract_year"] = pick(raw, ["dealYear"]).map(to_number)
    out["contract_month"] = pick(raw, ["dealMonth"]).map(to_number)
    out["contract_day"] = pick(raw, ["dealDay"]).map(to_number)
    out["contract_date"] = pd.to_datetime(
        {"year": out["contract_year"], "month": out["contract_month"], "day": out["contract_day"]},
        errors="coerce",
    )
    out["ym"] = out["contract_date"].dt.to_period("M").astype("string")
    out["deposit_won"] = pick(raw, ["deposit"]).map(to_number) * 10_000
    out["monthly_rent_won"] = pick(raw, ["monthlyRent"]).map(to_number) * 10_000
    out["area_m2"] = pick(raw, ["excluUseAr", "totalFloorAr", "전용면적", "연면적"]).map(to_number)
    out["floor"] = pick(raw, ["floor"]).map(to_number)
    out["build_year"] = pick(raw, ["buildYear"]).map(to_number)
    out["rent_type"] = np.where(out["monthly_rent_won"].fillna(0) == 0, "jeonse", "monthly")
    out["deposit_per_m2"] = out["deposit_won"] / out["area_m2"].replace(0, np.nan)
    out["monthly_rent_per_m2"] = out["monthly_rent_won"] / out["area_m2"].replace(0, np.nan)
    out["contract_term"] = pick(raw, ["contractTerm"])
    out["contract_type"] = pick(raw, ["contractType"])
    out = out.merge(gu_map, on="gu_code", how="left")
    return out[(out["ym"] >= START_YM) & (out["ym"] <= END_YM)].copy()


def clean_sale(gu_map: pd.DataFrame) -> pd.DataFrame:
    raw = read_rtms_files("sale")
    if raw.empty:
        return raw

    out = pd.DataFrame(index=raw.index)
    out["trade_type"] = "sale"
    out["property_type"] = pick(raw, ["property_type"])
    out["gu_code"] = pick(raw, ["sggCd", "request_gu_code"]).astype("string").str[:5]
    out["dong_name"] = pick(raw, ["umdNm"])
    out["building_name"] = pick(raw, ["aptNm", "mhouseNm", "offiNm"])
    out["house_type"] = pick(raw, ["houseType"])
    out["jibun"] = pick(raw, ["jibun"])
    out["contract_year"] = pick(raw, ["dealYear"]).map(to_number)
    out["contract_month"] = pick(raw, ["dealMonth"]).map(to_number)
    out["contract_day"] = pick(raw, ["dealDay"]).map(to_number)
    out["contract_date"] = pd.to_datetime(
        {"year": out["contract_year"], "month": out["contract_month"], "day": out["contract_day"]},
        errors="coerce",
    )
    out["ym"] = out["contract_date"].dt.to_period("M").astype("string")
    out["sale_price_won"] = pick(raw, ["dealAmount"]).map(to_number) * 10_000
    out["area_m2"] = pick(raw, ["excluUseAr", "totalFloorAr", "buildingAr", "전용면적", "연면적"]).map(to_number)
    out["land_area_m2"] = pick(raw, ["plottageAr", "landAr", "대지면적"]).map(to_number)
    out["floor"] = pick(raw, ["floor"]).map(to_number)
    out["build_year"] = pick(raw, ["buildYear"]).map(to_number)
    out["sale_price_per_m2"] = out["sale_price_won"] / out["area_m2"].replace(0, np.nan)
    out = out.merge(gu_map, on="gu_code", how="left")
    return out[(out["ym"] >= START_YM) & (out["ym"] <= END_YM)].copy()


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    gu_map = make_gu_map()

    gu_map.to_csv(CLEAN_DIR / "gu_map_seoul.csv", index=False, encoding="utf-8-sig")
    clean_lh(gu_map).to_csv(CLEAN_DIR / "lh_clean.csv", index=False, encoding="utf-8-sig")
    clean_rent(gu_map).to_csv(CLEAN_DIR / "rent_clean.csv", index=False, encoding="utf-8-sig")
    clean_sale(gu_map).to_csv(CLEAN_DIR / "sale_clean.csv", index=False, encoding="utf-8-sig")
    print(f"saved clean files to {CLEAN_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
