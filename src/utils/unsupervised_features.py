import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import DATA_PATH, ROOT, TARGET, load_training_data, make_preprocessor, make_splits, select_features


RESULT_DIR = ROOT / "res" / "unsupervised_features"
OUTPUT_PATH = ROOT / "data" / "modeling" / "modeling_dataset_unsupervised.csv"
DEFAULT_METHODS = ["kmeans", "isolation_forest"]
MAD_SCALE = 1.4826
EPS = 1e-9


def to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def add_kmeans_scores(
    out: pd.DataFrame,
    x_train: np.ndarray,
    x_all: np.ndarray,
    train_mask: pd.Series,
    n_clusters: int,
    min_cluster_size: int,
    min_mad: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    kmeans.fit(x_train)
    cluster_id = kmeans.predict(x_all)
    distances = kmeans.transform(x_all)
    cluster_distance = distances[np.arange(len(out)), cluster_id]

    train_cluster_id = cluster_id[train_mask.to_numpy()]
    train_distance = cluster_distance[train_mask.to_numpy()]
    global_median = float(np.median(train_distance))
    global_mad = float(np.median(np.abs(train_distance - global_median)))
    global_mad = max(global_mad, min_mad)
    cluster_stats = {}
    for cluster in range(n_clusters):
        values = train_distance[train_cluster_id == cluster]
        use_global_fallback = len(values) < min_cluster_size
        if len(values) == 0:
            median = global_median
            mad = global_mad
        else:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
        if mad < min_mad:
            use_global_fallback = True
        if use_global_fallback:
            median = global_median
            mad = global_mad
        cluster_stats[cluster] = {
            "distance_median": median,
            "distance_mad": mad,
            "train_rows": int(len(values)),
            "used_global_fallback": bool(use_global_fallback),
        }

    robust_z = np.zeros(len(out), dtype=float)
    for cluster in range(n_clusters):
        mask = cluster_id == cluster
        median = cluster_stats[cluster]["distance_median"]
        mad = cluster_stats[cluster]["distance_mad"]
        robust_z[mask] = (cluster_distance[mask] - median) / (MAD_SCALE * mad + EPS)

    out["cluster_id"] = cluster_id.astype(str)
    out["kmeans_distance"] = cluster_distance
    out["kmeans_robust_z"] = robust_z
    out["cluster_distance"] = cluster_distance
    out["robust_z"] = robust_z

    report = {
        "n_clusters": n_clusters,
        "min_cluster_size": min_cluster_size,
        "min_mad": min_mad,
        "global_distance_median": global_median,
        "global_distance_mad": global_mad,
        "cluster_counts": pd.Series(cluster_id).value_counts().sort_index().astype(int).to_dict(),
        "cluster_stats": cluster_stats,
        "new_columns": ["cluster_id", "kmeans_distance", "kmeans_robust_z", "cluster_distance", "robust_z"],
    }
    return out, report


def add_isolation_forest_score(
    out: pd.DataFrame,
    x_train: np.ndarray,
    x_all: np.ndarray,
    contamination: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    isolation = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    isolation.fit(x_train)
    isolation_score = -isolation.decision_function(x_all)
    out["isolation_score"] = isolation_score
    return out, {
        "contamination": contamination,
        "new_columns": ["isolation_score"],
    }


def add_lof_score(
    out: pd.DataFrame,
    x_train: np.ndarray,
    x_all: np.ndarray,
    n_neighbors: int,
    contamination: float,
) -> tuple[pd.DataFrame, dict]:
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(x_train)
    out["lof_score"] = -lof.decision_function(x_all)
    return out, {
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "new_columns": ["lof_score"],
    }


def add_autoencoder_error(
    out: pd.DataFrame,
    x_train: np.ndarray,
    x_all: np.ndarray,
    hidden_layer_sizes: tuple[int, ...],
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_all_scaled = scaler.transform(x_all)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        random_state=random_state,
    )
    autoencoder.fit(x_train_scaled, x_train_scaled)
    reconstructed = autoencoder.predict(x_all_scaled)
    out["autoencoder_error"] = np.mean((x_all_scaled - reconstructed) ** 2, axis=1)
    return out, {
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "max_iter": autoencoder.max_iter,
        "new_columns": ["autoencoder_error"],
    }


def fit_transform_features(
    df: pd.DataFrame,
    target: str,
    train_mask: pd.Series,
    methods: list[str],
    n_clusters: int,
    contamination: float,
    lof_neighbors: int,
    autoencoder_hidden: tuple[int, ...],
    min_cluster_size: int,
    min_mad: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    features, cat_cols, num_cols = select_features(df, target)
    preprocessor = make_preprocessor(cat_cols, num_cols)

    x_train = preprocessor.fit_transform(df.loc[train_mask, features])
    x_all = preprocessor.transform(df[features])
    x_train = to_dense(x_train)
    x_all = to_dense(x_all)

    out = df.copy()
    method_reports = {}
    if "kmeans" in methods:
        out, method_reports["kmeans"] = add_kmeans_scores(
            out,
            x_train,
            x_all,
            train_mask,
            n_clusters,
            min_cluster_size,
            min_mad,
            random_state,
        )
    if "isolation_forest" in methods:
        out, method_reports["isolation_forest"] = add_isolation_forest_score(
            out,
            x_train,
            x_all,
            contamination,
            random_state,
        )
    if "lof" in methods:
        out, method_reports["lof"] = add_lof_score(out, x_train, x_all, lof_neighbors, contamination)
    if "autoencoder" in methods:
        out, method_reports["autoencoder"] = add_autoencoder_error(
            out,
            x_train,
            x_all,
            autoencoder_hidden,
            random_state,
        )

    score_cols = [
        col
        for col in ["kmeans_distance", "kmeans_robust_z", "cluster_distance", "robust_z", "isolation_score", "lof_score", "autoencoder_error"]
        if col in out.columns
    ]

    report = {
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "methods": methods,
        "random_state": random_state,
        "feature_count": len(features),
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "method_reports": method_reports,
        "new_columns": [col for col in ["cluster_id"] + score_cols if col in out.columns],
        "score_summary": {
            col: out[col].describe().to_dict()
            for col in score_cols
        },
    }
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clustering/anomaly features without target leakage.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["kmeans", "isolation_forest", "lof", "autoencoder"],
        default=DEFAULT_METHODS,
        help="Unsupervised methods to fit and append as score columns.",
    )
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-mad", type=float, default=1e-3)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--lof-neighbors", type=int, default=35)
    parser.add_argument("--autoencoder-hidden", nargs="+", type=int, default=[64, 16, 64])
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_dir = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(Path(args.data), args.target)
    train_mask, valid_mask, test_mask, split_report = make_splits(
        df,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    out, report = fit_transform_features(
        df=df,
        target=args.target,
        train_mask=train_mask,
        methods=args.methods,
        n_clusters=args.n_clusters,
        contamination=args.contamination,
        lof_neighbors=args.lof_neighbors,
        autoencoder_hidden=tuple(args.autoencoder_hidden),
        min_cluster_size=args.min_cluster_size,
        min_mad=args.min_mad,
        random_state=args.random_state,
    )
    report["split"] = split_report
    report["split_score_summary"] = {}
    score_cols = [col for col in report["score_summary"] if col in out.columns]
    for name, mask in [("train", train_mask), ("valid", valid_mask), ("test", test_mask)]:
        sub = out.loc[mask, score_cols]
        report["split_score_summary"][name] = {
            col: sub[col].describe().to_dict()
            for col in sub.columns
        }

    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_path.relative_to(ROOT)} rows={len(out):,}")
    print(f"saved {report_path.relative_to(ROOT)}")
    print(f"new columns: {', '.join(report['new_columns'])}")


if __name__ == "__main__":
    main()
