import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import ROOT, prediction_suffix


DEFAULT_ARTIFACT_DIR = ROOT / "res" / "final_model"
MISSING_VALUE = "__MISSING__"

IN_DOMAIN_PROPERTY_TYPES = {"apt", "officetel", "rh", "detached"}
OOD_PROPERTY_LABELS = {
    "urban": "도시형생활주택",
    "other": "복합/기타 주택",
}


def model_scope_info(property_type: str) -> tuple[int, str, str, str]:
    if property_type in IN_DOMAIN_PROPERTY_TYPES:
        return (0, "양호", "in_domain", "학습 데이터에 포함된 주택유형입니다.")
    label = OOD_PROPERTY_LABELS.get(property_type, property_type or "알 수 없음")
    return (75, "주의", "out_of_domain", f"{label}은 직접 학습 범위 밖 유형입니다. 예측값은 유사 패턴 기준 참고값으로 해석하세요.")


def add_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    combo_specs = {
        "gu_property": ["gu_name", "property_type"],
        "gu_ym": ["gu_name", "ym"],
        "property_ym": ["property_type", "ym"],
        "housing_type_ym": ["주택유형", "ym"],
        "detail_type_ym": ["유형", "ym"],
    }
    for new_col, cols in combo_specs.items():
        if all(col in out.columns for col in cols):
            values = [out[col].astype("string").fillna(MISSING_VALUE) for col in cols]
            out[new_col] = values[0]
            for value in values[1:]:
                out[new_col] = out[new_col] + "__" + value
    return out


def prepare_catboost_frame(df: pd.DataFrame, features: list[str], cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = np.nan
    out = out[features].copy()
    for col in cat_cols:
        out[col] = out[col].astype("string").fillna(MISSING_VALUE)
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_json:
        payload = json.loads(args.input_json)
        if isinstance(payload, dict):
            payload = [payload]
        return pd.DataFrame(payload)
    if args.input_file:
        path = Path(args.input_file)
        if path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(path)
        return pd.read_csv(path, low_memory=False)
    sample = {
        "gu_name": "강남구",
        "gu_code": "11680",
        "ym": "2025-01",
        "property_type": "apt",
        "주택유형": "아파트",
        "유형": "아파트",
        "area_m2_clean": 59.0,
        "room_count_clean": 3,
        "household_size_clean": 2,
        "rent_deposit_median": 300000000,
        "deposit_won": 330000000,
    }
    return pd.DataFrame([sample])


def percentile_rank(value: float, reference) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0 or not np.isfinite(value):
        return 0.0
    return float(np.mean(reference <= value))

def build_unsupervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rent = pd.to_numeric(out.get("rent_deposit_median"), errors="coerce").clip(lower=0)
    out["rent_deposit_median_log"] = np.log1p(rent)
    return out


def calibrated_tail_score(value: float, reference, normal_quantile: float = 0.85, high_quantile: float = 0.999) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0 or not np.isfinite(value):
        return 0.0
    normal = float(np.quantile(reference, normal_quantile))
    high = float(np.quantile(reference, high_quantile))
    if high <= normal:
        return max(0.0, min(100.0, 100.0 * percentile_rank(value, reference)))
    return float(np.clip((value - normal) / (high - normal) * 100.0, 0.0, 100.0))


def _norm_key(value) -> str:
    if pd.isna(value):
        return MISSING_VALUE
    return str(value)


def _lookup_records(records: list[dict], values: list[str]) -> float | None:
    for record in records:
        if record.get("key") == values:
            return float(record["median"] if "median" in record else record["q"])
    return None


SOURCE_NAME_KO = {
    "gu_name+property_type+ym": "지역+주택구분+계약월",
    "gu_name+property_type": "지역+주택구분",
    "property_type+ym": "주택구분+계약월",
    "property_type": "주택구분",
    "gu_name": "지역",
    "global": "전체 중위값",
    "user_input": "사용자 입력",
    "missing": "없음",
}


def source_name_ko(source: str) -> str:
    return SOURCE_NAME_KO.get(source, source)


def infer_rent_deposit_median(row: pd.Series, artifacts: dict) -> tuple[float | None, str]:
    lookup = artifacts.get("rent_median_lookup", {})
    for level in lookup.get("levels", []):
        group_cols = level.get("group_cols", [])
        values = [_norm_key(row.get(col)) for col in group_cols]
        value = _lookup_records(level.get("records", []), values)
        if value is not None:
            return value, source_name_ko("+".join(group_cols))
    global_value = lookup.get("global")
    if global_value is None:
        return None, "missing"
    return float(global_value), source_name_ko("global")


def interval_quantile_for_row(row: pd.Series, metadata: dict) -> tuple[float, str]:
    lookup = metadata.get("interval_lookup") or {}
    for level in lookup.get("levels", []):
        group_cols = level.get("group_cols", [])
        values = [_norm_key(row.get(col)) for col in group_cols]
        value = _lookup_records(level.get("records", []), values)
        if value is not None:
            return value, source_name_ko("+".join(group_cols))
    return float(lookup.get("global_q", metadata["conformal_log_abs_residual_quantile"])), source_name_ko("global")


def score_grade(score: float) -> str:
    if score >= 80:
        return "매우 높음"
    if score >= 60:
        return "높음"
    if score >= 40:
        return "보통"
    if score >= 20:
        return "낮음"
    return "매우 낮음"


def risk_label(score: float) -> str:
    if score >= 80:
        return "high_risk"
    if score >= 60:
        return "caution"
    if score >= 40:
        return "fair"
    return "safe"


def add_unsupervised_scores(df: pd.DataFrame, metadata: dict, artifacts: dict) -> pd.DataFrame:
    out = df.copy()
    unsup = artifacts["unsupervised"]
    features = unsup["features"]
    cat_cols = unsup["cat_cols"]
    num_cols = unsup["num_cols"]
    unsup_input = build_unsupervised_frame(out)
    prepared = prepare_catboost_frame(add_combo_features(unsup_input), features, cat_cols, num_cols)
    x = unsup["preprocessor"].transform(prepared[features])
    if hasattr(x, "toarray"):
        x = x.toarray()

    raw_cluster_id = unsup["kmeans"].predict(x)
    cluster_mapping = unsup.get("cluster_mapping", {})
    internal_cluster_id = np.array([cluster_mapping.get(str(cluster), str(cluster)) for cluster in raw_cluster_id])
    display_mapping = unsup.get("cluster_display_mapping", {})
    cluster_id = np.array([display_mapping.get(str(cluster), str(cluster)) for cluster in internal_cluster_id])
    distance_matrix = unsup["kmeans"].transform(x)
    distance = distance_matrix[np.arange(len(out)), internal_cluster_id.astype(int)]
    isolation_score = -unsup["isolation_forest"].decision_function(x)
    out["raw_cluster_id"] = raw_cluster_id.astype(str)
    out["internal_cluster_id"] = internal_cluster_id.astype(str)
    out["cluster_id"] = cluster_id.astype(str)
    out["kmeans_distance"] = distance
    out["isolation_score"] = isolation_score
    refs = unsup["risk_reference"]
    out["kmeans_distance_percentile"] = [100 * percentile_rank(v, refs["kmeans_distance"]) for v in distance]
    out["isolation_score_percentile"] = [100 * percentile_rank(v, refs["isolation_score"]) for v in isolation_score]
    out["kmeans_tail_score"] = [calibrated_tail_score(v, refs["kmeans_distance"]) for v in distance]
    out["isolation_tail_score"] = [calibrated_tail_score(v, refs["isolation_score"]) for v in isolation_score]
    out["anomaly_raw_percentile"] = out[["kmeans_distance_percentile", "isolation_score_percentile"]].mean(axis=1)
    out["anomaly_percentile"] = (
        0.55 * out["kmeans_tail_score"] + 0.45 * out["isolation_tail_score"]
    ).clip(0, 100)
    plot_coordinates = unsup.get("plot_coordinates", {})
    if plot_coordinates.get("x") == "area_m2_clean" and plot_coordinates.get("y") == "rent_deposit_median_eok":
        out["cluster_x"] = pd.to_numeric(out.get("area_m2_clean"), errors="coerce")
        out["cluster_y"] = pd.to_numeric(out.get("rent_deposit_median"), errors="coerce") / 100_000_000
    elif "cluster_plot_features" in unsup and "cluster_plot_xy" in unsup:
        ref_x = np.asarray(unsup["cluster_plot_features"], dtype=float)
        ref_xy = np.asarray(unsup["cluster_plot_xy"], dtype=float)
        coords = []
        for row_x in x:
            distances = np.linalg.norm(ref_x - row_x, axis=1)
            k = min(12, len(distances))
            nearest = np.argpartition(distances, k - 1)[:k]
            weights = 1.0 / (distances[nearest] + 1e-6)
            weights = weights / weights.sum()
            coords.append((ref_xy[nearest] * weights[:, None]).sum(axis=0))
        coords = np.asarray(coords)
        out["cluster_x"] = coords[:, 0]
        out["cluster_y"] = coords[:, 1]
    elif "pca" in unsup:
        xy = unsup["pca"].transform(x)
        out["cluster_x"] = xy[:, 0]
        out["cluster_y"] = xy[:, 1]
    else:
        out["cluster_x"] = pd.to_numeric(unsup_input.get("area_m2_clean"), errors="coerce")
        out["cluster_y"] = pd.to_numeric(unsup_input.get("rent_deposit_median_log"), errors="coerce")
    return out


def predict(df: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "catboost_final.cbm"
    artifacts_path = artifact_dir / "artifacts.joblib"
    if not metadata_path.exists() or not model_path.exists() or not artifacts_path.exists():
        raise FileNotFoundError(
            f"final model artifacts not found under {artifact_dir}. Run src/models/train_final_model.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = joblib.load(artifacts_path)
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(model_path)

    features = metadata["features"]
    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]

    out = df.copy()
    if "rent_deposit_median" not in out.columns:
        out["rent_deposit_median"] = np.nan
    rent_sources = []
    for idx, row in out.iterrows():
        current = pd.to_numeric(row.get("rent_deposit_median"), errors="coerce")
        if pd.isna(current):
            inferred, source = infer_rent_deposit_median(row, artifacts)
            out.at[idx, "rent_deposit_median"] = inferred
            rent_sources.append(source)
        else:
            rent_sources.append(source_name_ko("user_input"))
    out["rent_deposit_median_source"] = rent_sources

    # rent_deposit_median is one of the model features, so it must be filled
    # before prediction. Otherwise web inputs without this field are treated
    # as missing and CatBoost can produce unrealistically large ratios.
    x = prepare_catboost_frame(add_combo_features(out), features, cat_cols, num_cols)
    pred_log_ratio = model.predict(x)

    interval_q = []
    interval_sources = []
    for _, row in out.iterrows():
        q_value, source = interval_quantile_for_row(row, metadata)
        interval_q.append(q_value)
        interval_sources.append(source)
    interval_q = np.asarray(interval_q, dtype=float)

    out["predicted_log_ratio"] = pred_log_ratio
    out["predicted_ratio"] = np.expm1(pred_log_ratio)
    out["predicted_ratio_lower_95"] = np.expm1(pred_log_ratio - interval_q)
    out["predicted_ratio_upper_95"] = np.expm1(pred_log_ratio + interval_q)
    out["interval_source"] = interval_sources

    out["predicted_fair_deposit_won"] = out["rent_deposit_median"] * out["predicted_ratio"]
    out["fair_deposit_lower_95_won"] = out["rent_deposit_median"] * out["predicted_ratio_lower_95"]
    out["fair_deposit_upper_95_won"] = out["rent_deposit_median"] * out["predicted_ratio_upper_95"]

    out = add_unsupervised_scores(out, metadata, artifacts)

    if "deposit_won" in out.columns:
        asking = pd.to_numeric(out["deposit_won"], errors="coerce")
        upper = out["fair_deposit_upper_95_won"].replace(0, np.nan)
        fair = out["predicted_fair_deposit_won"].replace(0, np.nan)
        out["deposit_gap_won"] = asking - out["predicted_fair_deposit_won"]
        out["deposit_gap_ratio"] = asking / fair
        fair_ratio = asking / fair
        upper_ratio = asking / upper
        high_price_score = ((fair_ratio - 1.05) / 0.25 * 100).clip(0, 100)
        low_price_score = ((0.85 - fair_ratio) / 0.30 * 100).clip(0, 100)
        upper_based = ((upper_ratio - 1.00) / 0.20 * 100).clip(0, 100)
        out["high_price_score"] = high_price_score.fillna(0.0)
        out["low_price_score"] = low_price_score.fillna(0.0)
        out["overprice_score"] = np.maximum.reduce([
            out["high_price_score"].to_numpy(),
            out["low_price_score"].to_numpy(),
            upper_based.fillna(0.0).to_numpy(),
        ])
    else:
        out["deposit_gap_won"] = np.nan
        out["deposit_gap_ratio"] = np.nan
        out["overprice_score"] = 0.0

    scope_records = [model_scope_info(str(value)) for value in out.get("property_type", pd.Series([""] * len(out)))]
    out["model_scope_score"] = [record[0] for record in scope_records]
    out["model_scope_label"] = [record[1] for record in scope_records]
    out["model_scope_status"] = [record[2] for record in scope_records]
    out["model_scope_message"] = [record[3] for record in scope_records]

    out["risk_probability_percent"] = out["overprice_score"].clip(0, 100)
    out["price_risk_grade"] = out["overprice_score"].map(score_grade)
    out["anomaly_grade"] = out["anomaly_percentile"].map(score_grade)
    out["risk_label"] = out["risk_probability_percent"].map(risk_label)
    out["confidence_level"] = metadata.get("confidence_level", 0.95)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fair jeonse deposit and risk index from final model artifacts.")
    parser.add_argument("--input-json", help="JSON object/list containing feature values.")
    parser.add_argument("--input-file", help="CSV or JSON file containing feature values.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = read_input(args)
    out = predict(df, Path(args.artifact_dir))
    display_cols = [
        col
        for col in [
            "gu_name",
            "ym",
            "property_type",
            "area_m2_clean",
            "room_count_clean",
            "rent_deposit_median",
            "deposit_won",
            "predicted_fair_deposit_won",
            "fair_deposit_lower_95_won",
            "fair_deposit_upper_95_won",
            "risk_probability_percent",
            "risk_label",
            "model_scope_label",
            "model_scope_message",
        ]
        if col in out.columns
    ]
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"saved {output_path}")
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
