import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) in sys.path:
    sys.path.remove(str(MODEL_DIR))

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import DATA_PATH, ROOT, TARGET, load_training_data, make_preprocessor, make_splits, metrics
from utils.scoreboard import metric_row, update_score_csv
from models.catboost_optuna import add_combo_features, prepare_catboost_frame, fit_catboost_model, MODEL_NAME


RESULT_DIR = ROOT / "res" / "final_model"
SCORE_PATH = ROOT / "res" / "score.csv"
UNSUPERVISED_SCORE_COLUMNS = ["kmeans_distance", "robust_z", "isolation_score"]
MISSING_VALUE = "__MISSING__"


def load_best_catboost_config(score_path: Path) -> tuple[pd.Series, dict]:
    score = pd.read_csv(score_path, low_memory=False)
    score = score[
        score["split"].eq("test")
        & score["experiment"].eq("catboost_optuna")
        & score["model"].eq("catboost_optuna")
    ].copy()
    if score.empty:
        raise ValueError("catboost_optuna test score not found in score.csv. Run catboost_optuna.py first.")
    best_row = score.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False]).iloc[0]
    metrics_path = ROOT / str(best_row["result_path"])
    if not metrics_path.exists():
        raise FileNotFoundError(f"best metrics file not found: {metrics_path}")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = report.get("study", {}).get("best_params")
    if not params:
        raise ValueError(f"best_params not found in {metrics_path}")
    params = dict(params)
    params.update(
        {
            "loss_function": report.get("study", {}).get("loss_function", "RMSE"),
            "eval_metric": "RMSE",
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": False,
        }
    )
    return best_row, params


def build_feature_lists(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    from utils.modeling import select_features

    features, cat_cols, num_cols = select_features(df, target)
    combo_cols = [
        col
        for col in ["gu_property", "gu_ym", "property_ym", "housing_type_ym", "detail_type_ym"]
        if col in df.columns and col not in cat_cols
    ]
    cat_cols = cat_cols + combo_cols
    features = [col for col in features if col not in combo_cols] + combo_cols
    num_cols = [col for col in features if col not in cat_cols and pd.api.types.is_numeric_dtype(df[col])]
    return features, cat_cols, num_cols


def build_unsupervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rent = pd.to_numeric(out.get("rent_deposit_median"), errors="coerce").clip(lower=0)
    out["rent_deposit_median_log"] = np.log1p(rent)
    return out


def build_unsupervised_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    # Clustering is for user-facing similarity/risk, not supervised price fitting.
    # Use broad, stable fields only. Keeping gu_name out avoids rare district/type
    # combinations being treated as extreme anomalies too easily.
    cat_cols = [col for col in ["property_type"] if col in df.columns]
    num_candidates = ["area_m2_clean", "room_count_clean", "rent_deposit_median_log"]
    num_cols = [col for col in num_candidates if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    features = cat_cols + num_cols
    if not features:
        raise ValueError("no usable unsupervised clustering features found")
    return features, cat_cols, num_cols


def fit_unsupervised_artifacts(
    df: pd.DataFrame,
    target: str,
    train_mask: pd.Series,
    random_state: int,
    n_clusters: int = 6,
    min_cluster_rows: int = 500,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest

    unsup_df = build_unsupervised_frame(df)
    features, cat_cols, num_cols = build_unsupervised_feature_lists(unsup_df)
    preprocessor = make_preprocessor(cat_cols, num_cols)
    x_train = preprocessor.fit_transform(unsup_df.loc[train_mask, features])
    if hasattr(x_train, "toarray"):
        x_train_dense = x_train.toarray()
    else:
        x_train_dense = x_train

    kmeans = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
    kmeans.fit(x_train_dense)
    raw_train_cluster = kmeans.predict(x_train_dense)
    all_distances = kmeans.transform(x_train_dense)
    cluster_counts = pd.Series(raw_train_cluster).value_counts().sort_index().to_dict()
    large_clusters = sorted([cluster for cluster, count in cluster_counts.items() if count >= min_cluster_rows])
    if not large_clusters:
        large_clusters = sorted(cluster_counts, key=cluster_counts.get, reverse=True)[: max(1, min(3, len(cluster_counts)))]
    cluster_mapping = {}
    for cluster in range(n_clusters):
        if cluster in large_clusters:
            cluster_mapping[str(cluster)] = str(cluster)
            continue
        nearest_large = min(large_clusters, key=lambda large: float(np.linalg.norm(kmeans.cluster_centers_[cluster] - kmeans.cluster_centers_[large])))
        cluster_mapping[str(cluster)] = str(nearest_large)
    train_cluster = np.array([int(cluster_mapping[str(cluster)]) for cluster in raw_train_cluster])
    merged_labels = sorted({str(cluster) for cluster in train_cluster}, key=int)
    cluster_display_mapping = {label: str(idx + 1) for idx, label in enumerate(merged_labels)}
    train_distance = all_distances[np.arange(len(x_train_dense)), train_cluster]

    display_train_cluster = np.array([cluster_display_mapping[str(cluster)] for cluster in train_cluster])
    rng = np.random.default_rng(random_state)
    plot_indices = []
    for label in sorted(set(display_train_cluster), key=int):
        idx = np.where(display_train_cluster == label)[0]
        take = min(350, len(idx))
        if take > 0:
            plot_indices.extend(rng.choice(idx, size=take, replace=False).tolist())
    plot_indices = sorted(plot_indices)
    plot_df = df.loc[train_mask].iloc[plot_indices].copy()
    plot_area = pd.to_numeric(plot_df.get("area_m2_clean"), errors="coerce").to_numpy(dtype=float)
    plot_rent_eok = (pd.to_numeric(plot_df.get("rent_deposit_median"), errors="coerce") / 100_000_000).to_numpy(dtype=float)
    area_jitter = rng.normal(0.0, 0.35, size=len(plot_indices))
    rent_jitter = rng.normal(0.0, 0.025, size=len(plot_indices))
    cluster_plot_points = [
        {
            "x": float(plot_area[pos] + area_jitter[pos]),
            "y": float(max(0.0, plot_rent_eok[pos] + rent_jitter[pos])),
            "cluster_id": str(display_train_cluster[idx]),
        }
        for pos, idx in enumerate(plot_indices)
        if np.isfinite(plot_area[pos]) and np.isfinite(plot_rent_eok[pos])
    ]

    isolation = IsolationForest(n_estimators=300, contamination=0.02, random_state=random_state, n_jobs=-1)
    isolation.fit(x_train_dense)
    train_isolation_score = -isolation.decision_function(x_train_dense)

    risk_reference = {
        "kmeans_distance": train_distance,
        "isolation_score": train_isolation_score,
    }
    return {
        "features": features,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "isolation_forest": isolation,
        "cluster_plot_points": cluster_plot_points,
        "plot_coordinates": {"x": "area_m2_clean", "y": "rent_deposit_median_eok"},
        "risk_reference": risk_reference,
        "n_clusters": n_clusters,
        "min_cluster_rows": min_cluster_rows,
        "raw_cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "large_clusters": [str(cluster) for cluster in large_clusters],
        "cluster_mapping": cluster_mapping,
        "cluster_display_mapping": cluster_display_mapping,
    }


def percentile_rank(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return 0.0
    return float(np.mean(reference <= value))


def _string_key(value) -> str:
    if pd.isna(value):
        return MISSING_VALUE
    return str(value)


def _median_records(df: pd.DataFrame, group_cols: list[str], value_col: str) -> list[dict]:
    cols = group_cols + [value_col]
    sub = df[cols].dropna(subset=[value_col]).copy()
    if sub.empty:
        return []
    grouped = sub.groupby(group_cols, dropna=False)[value_col].agg(["median", "count"]).reset_index()
    records = []
    for row in grouped.to_dict("records"):
        records.append(
            {
                "key": [_string_key(row[col]) for col in group_cols],
                "median": float(row["median"]),
                "count": int(row["count"]),
            }
        )
    return records


def build_rent_median_lookup(df: pd.DataFrame) -> dict:
    value_col = "rent_deposit_median"
    if value_col not in df.columns:
        return {"global": None, "levels": []}
    valid = df[df[value_col].notna()].copy()
    levels = []
    specs = [
        ["gu_name", "property_type", "ym"],
        ["gu_name", "property_type"],
        ["property_type", "ym"],
        ["property_type"],
        ["gu_name"],
    ]
    for group_cols in specs:
        missing = [col for col in group_cols if col not in valid.columns]
        if missing:
            continue
        levels.append({"group_cols": group_cols, "records": _median_records(valid, group_cols, value_col)})
    return {
        "global": float(valid[value_col].median()) if len(valid) else None,
        "levels": levels,
    }


def _residual_quantile_records(
    valid_df: pd.DataFrame,
    residual_abs: np.ndarray,
    group_cols: list[str],
    alpha: float,
    min_rows: int,
) -> list[dict]:
    missing = [col for col in group_cols if col not in valid_df.columns]
    if missing:
        return []
    sub = valid_df[group_cols].copy()
    sub["_residual_abs"] = residual_abs
    records = []
    for key, group in sub.groupby(group_cols, dropna=False):
        if len(group) < min_rows:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        records.append(
            {
                "key": [_string_key(value) for value in key],
                "q": float(np.quantile(group["_residual_abs"].to_numpy(), 1.0 - alpha)),
                "count": int(len(group)),
            }
        )
    return records


def build_interval_lookup(valid_df: pd.DataFrame, residual_abs: np.ndarray, alpha: float, min_rows: int = 30) -> dict:
    levels = []
    specs = [
        ["gu_name", "property_type"],
        ["property_type"],
        ["gu_name"],
    ]
    for group_cols in specs:
        records = _residual_quantile_records(valid_df, residual_abs, group_cols, alpha, min_rows)
        if records:
            levels.append({"group_cols": group_cols, "records": records})
    return {
        "alpha": alpha,
        "min_rows": min_rows,
        "global_q": float(np.quantile(residual_abs, 1.0 - alpha)),
        "levels": levels,
    }


def build_cluster_profile(df: pd.DataFrame, unsupervised: dict, train_mask: pd.Series) -> dict:
    features = unsupervised["features"]
    cat_cols = unsupervised["cat_cols"]
    num_cols = unsupervised["num_cols"]
    unsup_df = build_unsupervised_frame(df)
    prepared = prepare_catboost_frame(unsup_df.loc[train_mask], features, cat_cols, num_cols)
    x = unsupervised["preprocessor"].transform(prepared[features])
    if hasattr(x, "toarray"):
        x = x.toarray()
    raw_cluster_id = unsupervised["kmeans"].predict(x)
    cluster_mapping = unsupervised.get("cluster_mapping", {})
    internal_cluster_id = np.array([cluster_mapping.get(str(cluster), str(cluster)) for cluster in raw_cluster_id])
    display_mapping = unsupervised.get("cluster_display_mapping", {})
    cluster_id = np.array([display_mapping.get(str(cluster), str(cluster)) for cluster in internal_cluster_id])
    sub = df.loc[train_mask].copy()
    sub["cluster_id"] = cluster_id.astype(str)
    profiles = []
    for cluster, group in sub.groupby("cluster_id"):
        top_gu = group["gu_name"].value_counts().head(3).to_dict() if "gu_name" in group else {}
        top_property = group["property_type"].value_counts().head(3).to_dict() if "property_type" in group else {}
        profiles.append(
            {
                "cluster_id": str(cluster),
                "rows": int(len(group)),
                "avg_area_m2": float(group["area_m2_clean"].mean()) if "area_m2_clean" in group else None,
                "median_rent_deposit_won": float(group["rent_deposit_median"].median()) if "rent_deposit_median" in group else None,
                "median_deposit_won": float(group["deposit_won"].median()) if "deposit_won" in group else None,
                "top_gu_name": top_gu,
                "top_property_type": top_property,
            }
        )
    return {"profiles": sorted(profiles, key=lambda item: int(item["cluster_id"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final fair-deposit and risk prediction artifacts.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--score-path", default=str(SCORE_PATH))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for conformal interval.")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--min-cluster-rows", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_row, best_params = load_best_catboost_config(Path(args.score_path))
    df = load_training_data(Path(args.data), args.target)
    df = add_combo_features(df)
    features, cat_cols, num_cols = build_feature_lists(df, args.target)
    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_train = prepare_catboost_frame(df.loc[train_mask], features, cat_cols, num_cols)
    y_train = df.loc[train_mask, args.target].to_numpy()
    x_valid = prepare_catboost_frame(df.loc[valid_mask], features, cat_cols, num_cols)
    y_valid = df.loc[valid_mask, args.target].to_numpy()
    x_test = prepare_catboost_frame(df.loc[test_mask], features, cat_cols, num_cols)
    y_test = df.loc[test_mask, args.target].to_numpy()

    model = fit_catboost_model(x_train, y_train, x_valid, y_valid, cat_cols, best_params)
    valid_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)
    valid_scores = metrics(y_valid, valid_pred)
    test_scores = metrics(y_test, test_pred)

    residual_abs = np.abs(y_valid - valid_pred)
    interval_lookup = build_interval_lookup(df.loc[valid_mask].copy(), residual_abs, args.alpha)
    conformal_q = interval_lookup["global_q"]
    residual_signed = y_valid - valid_pred

    unsupervised = fit_unsupervised_artifacts(
        df,
        args.target,
        train_mask,
        args.random_state,
        n_clusters=args.n_clusters,
        min_cluster_rows=args.min_cluster_rows,
    )
    rent_median_lookup = build_rent_median_lookup(df)
    cluster_profile = build_cluster_profile(df, unsupervised, train_mask)

    model_path = output_dir / "catboost_final.cbm"
    model.save_model(model_path)
    artifacts_path = output_dir / "artifacts.joblib"
    joblib.dump(
        {
            "unsupervised": unsupervised,
            "valid_residual_abs": residual_abs,
            "valid_residual_signed": residual_signed,
            "rent_median_lookup": rent_median_lookup,
            "cluster_profile": cluster_profile,
        },
        artifacts_path,
    )

    metadata = {
        "selected_from_score": best_row.to_dict(),
        "target": args.target,
        "model_name": MODEL_NAME,
        "best_params": best_params,
        "split": split_report,
        "features": features,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "alpha": args.alpha,
        "confidence_level": 1.0 - args.alpha,
        "conformal_log_abs_residual_quantile": conformal_q,
        "interval_lookup": interval_lookup,
        "valid": valid_scores,
        "test": test_scores,
        "model_path": str(model_path.relative_to(ROOT)),
        "artifacts_path": str(artifacts_path.relative_to(ROOT)),
        "cluster_rule": {
            "n_clusters": args.n_clusters,
            "min_cluster_rows": args.min_cluster_rows,
            "small_clusters_are_merged_to_nearest_large_cluster": True,
        },
        "risk_rule": {
            "needs_deposit_won_for_overprice_risk": True,
            "risk_index": "0.65 * overprice_score + 0.35 * anomaly_percentile",
            "labels": {
                "0-39": "safe",
                "40-59": "fair",
                "60-79": "caution",
                "80-100": "high_risk",
            },
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    update_score_csv(
        Path(args.score_path),
        [
            metric_row(
                experiment="final_model",
                run_name=str(best_row.get("run_name", "catboost_optuna")),
                model="catboost_final",
                split="test",
                metrics=test_scores,
                target=args.target,
                result_path=metadata_path.relative_to(ROOT),
                rows=split_report["rows"]["test"],
                notes=f"source={best_row.get('experiment')}/{best_row.get('run_name')},alpha={args.alpha}",
            )
        ],
    )

    print(f"selected best model: {best_row['experiment']} {best_row['run_name']} {best_row['model']}")
    print(f"test metrics: {test_scores}")
    print(f"95% log interval half-width: {conformal_q:.6f}")
    print(f"saved {metadata_path.relative_to(ROOT)}")
    print(f"saved {model_path.relative_to(ROOT)}")
    print(f"saved {artifacts_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
