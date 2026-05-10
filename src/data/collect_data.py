import argparse
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
from requests import HTTPError


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RTMS_DIR = RAW_DIR / "rtms_api"

LAWD_PATH = RAW_DIR / "국토교통부_전국 법정동_20250807.csv"
LH_PATH = RAW_DIR / "한국토지주택공사_임대주택공급현황_전세임대_20251120.csv"

START_YM = "202201"
END_YM = "202512"
NUM_OF_ROWS = 1000
REQUEST_SLEEP_SEC = 0.05


@dataclass(frozen=True)
class RtmsApi:
    name: str
    trade_type: str
    property_type: str
    url: str


RTMS_APIS = {
    "rent_apt": RtmsApi(
        "rent_apt",
        "rent",
        "apt",
        "http://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
    ),
    "rent_rh": RtmsApi(
        "rent_rh",
        "rent",
        "rh",
        "http://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent",
    ),
    "rent_officetel": RtmsApi(
        "rent_officetel",
        "rent",
        "officetel",
        "http://apis.data.go.kr/1613000/RTMSDataSvcOffiRent/getRTMSDataSvcOffiRent",
    ),
    "rent_detached": RtmsApi(
        "rent_detached",
        "rent",
        "detached",
        "http://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent",
    ),
    "sale_apt": RtmsApi(
        "sale_apt",
        "sale",
        "apt",
        "http://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade",
    ),
    "sale_rh": RtmsApi(
        "sale_rh",
        "sale",
        "rh",
        "http://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade",
    ),
    "sale_officetel": RtmsApi(
        "sale_officetel",
        "sale",
        "officetel",
        "http://apis.data.go.kr/1613000/RTMSDataSvcOffiTrade/getRTMSDataSvcOffiTrade",
    ),
    "sale_detached": RtmsApi(
        "sale_detached",
        "sale",
        "detached",
        "http://apis.data.go.kr/1613000/RTMSDataSvcSHTrade/getRTMSDataSvcSHTrade",
    ),
}


def month_range(start_ym: str, end_ym: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(start=start_ym, end=end_ym, freq="M")]


def load_seoul_gu_codes() -> list[str]:
    lawd = pd.read_csv(LAWD_PATH, encoding="utf-8-sig")
    lawd = lawd[(lawd["시도명"] == "서울특별시") & lawd["삭제일자"].isna()].copy()
    gu = lawd[lawd["시군구명"].notna()].copy()
    return sorted(gu["법정동코드"].astype(str).str[:5].drop_duplicates().tolist())


def parse_xml_response(xml_text: str) -> tuple[list[dict], int]:
    root = ET.fromstring(xml_text)
    result_code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
    result_msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg")
    if result_code and result_code not in {"000", "00"}:
        raise RuntimeError(f"API error {result_code}: {result_msg}")

    total_count = int(root.findtext(".//totalCount") or 0)
    rows = [{child.tag: (child.text or "").strip() for child in item} for item in root.findall(".//item")]
    return rows, total_count


def fetch_page(
    session: requests.Session,
    api: RtmsApi,
    service_key: str,
    gu_code: str,
    deal_ymd: str,
    page_no: int,
) -> tuple[list[dict], int]:
    params = {
        "serviceKey": service_key,
        "LAWD_CD": gu_code,
        "DEAL_YMD": deal_ymd,
        "pageNo": page_no,
        "numOfRows": NUM_OF_ROWS,
    }
    response = session.get(api.url, params=params, timeout=30)
    try:
        response.raise_for_status()
    except HTTPError as exc:
        response_preview = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"{api.name} API 호출 실패: HTTP {response.status_code} "
            f"(LAWD_CD={gu_code}, DEAL_YMD={deal_ymd}). "
            "공공데이터포털에서 해당 API를 활용신청했는지, 인증키가 맞는지 확인하세요. "
            f"응답 내용: {response_preview}"
        ) from exc
    return parse_xml_response(response.text)


def fetch_month(session: requests.Session, api: RtmsApi, service_key: str, gu_code: str, deal_ymd: str) -> list[dict]:
    rows, total_count = fetch_page(session, api, service_key, gu_code, deal_ymd, 1)
    pages = int(np.ceil(total_count / NUM_OF_ROWS)) if total_count else 1
    for page_no in range(2, pages + 1):
        page_rows, _ = fetch_page(session, api, service_key, gu_code, deal_ymd, page_no)
        rows.extend(page_rows)
        time.sleep(REQUEST_SLEEP_SEC)
    return rows


def collect_api_targets(targets: list[str], gu_codes: list[str], months: list[str], service_key: str, overwrite: bool) -> None:
    RTMS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for target in targets:
        api = RTMS_APIS[target]
        out_path = RTMS_DIR / f"{api.name}_{months[0]}_{months[-1]}.csv"
        if out_path.exists() and not overwrite:
            print(f"skip existing {out_path.relative_to(ROOT)}")
            continue

        rows = []
        for gu_code in gu_codes:
            for deal_ymd in months:
                month_rows = fetch_month(session, api, service_key, gu_code, deal_ymd)
                for row in month_rows:
                    row["source_api"] = api.name
                    row["trade_type"] = api.trade_type
                    row["property_type"] = api.property_type
                    row["request_gu_code"] = gu_code
                    row["request_deal_ymd"] = deal_ymd
                rows.extend(month_rows)
                print(f"{api.name} {gu_code} {deal_ymd}: {len(month_rows):,}")
                time.sleep(REQUEST_SLEEP_SEC)

        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved {out_path.relative_to(ROOT)} rows={len(rows):,}")


def parse_targets(value: str) -> list[str]:
    if value == "all":
        return list(RTMS_APIS)
    if value == "rent":
        return [k for k, v in RTMS_APIS.items() if v.trade_type == "rent"]
    if value == "sale":
        return [k for k, v in RTMS_APIS.items() if v.trade_type == "sale"]
    if value == "detached":
        return [k for k, v in RTMS_APIS.items() if v.property_type == "detached"]
    targets = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(targets) - set(RTMS_APIS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="01. collect raw data")
    parser.add_argument("--start-ym", default=START_YM)
    parser.add_argument("--end-ym", default=END_YM)
    parser.add_argument("--targets", default="all", help="all, rent, sale, or comma-separated target names")
    parser.add_argument("--gu-codes", default="", help="default: all Seoul gu codes. example: 11110,11680")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    months = month_range(args.start_ym, args.end_ym)
    gu_codes = [x.strip() for x in args.gu_codes.split(",") if x.strip()] or load_seoul_gu_codes()
    targets = parse_targets(args.targets)

    if args.dry_run:
        print(f"period: {months[0]}~{months[-1]} ({len(months)} months)")
        print(f"gu_codes: {len(gu_codes)} Seoul gu codes")
        print(f"targets: {targets}")
        print(f"api requests before pagination: {len(months) * len(gu_codes) * len(targets):,}")
        return

    if args.skip_api:
        return

    service_key = os.environ.get("DATA_GO_KR_SERVICE_KEY")
    if not service_key:
        raise SystemExit("DATA_GO_KR_SERVICE_KEY 환경변수에 공공데이터포털 인증키를 넣어주세요.")
    collect_api_targets(targets, gu_codes, months, unquote(service_key), args.overwrite)


if __name__ == "__main__":
    main()
