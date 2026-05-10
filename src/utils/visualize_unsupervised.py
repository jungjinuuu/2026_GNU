import argparse
import json
import os
import sys
from pathlib import Path

Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)
Path("/tmp/fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import DATA_PATH, ROOT, TARGET, load_training_data, make_preprocessor, make_splits, select_features


UNSUPERVISED_DATA_PATH = ROOT / "data" / "modeling" / "modeling_dataset_unsupervised.csv"
RESULT_DIR = ROOT / "res" / "unsupervised_features" / "plots"
DEFAULT_SCORE_COLUMNS = [
    "cluster_distance",
    "robust_z",
    "isolation_score",
    "lof_score",
    "autoencoder_error",
]


def save_histograms(df: pd.DataFrame, score_cols: list[str], output_dir: Path, clip_quantile: float) -> list[str]:
    paths = []
    for col in score_cols:
        values = df[col].dropna()
        upper = values.quantile(clip_quantile)
        lower = values.quantile(1 - clip_quantile)
        clipped = values.clip(lower, upper)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(clipped, bins=60, color="#4c78a8", alpha=0.85)
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(f"{col} clipped to [{1 - clip_quantile:.4f}, {clip_quantile:.4f}] quantiles")
        ax.set_ylabel("count")
        fig.tight_layout()
        path = output_dir / f"hist_{col}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def save_target_scatter(df: pd.DataFrame, score_cols: list[str], target: str, output_dir: Path) -> list[str]:
    paths = []
    for col in score_cols:
        sample = df[[col, target]].dropna()
        if len(sample) > 15000:
            sample = sample.sample(15000, random_state=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sample[col], sample[target], s=5, alpha=0.25, color="#f58518")
        ax.set_title(f"{target} vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        fig.tight_layout()
        path = output_dir / f"scatter_target_{col}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def save_pca_plots(
    df: pd.DataFrame,
    data_path: Path,
    target: str,
    score_cols: list[str],
    output_dir: Path,
    sample_size: int,
) -> list[str]:
    source_df = load_training_data(data_path, target)
    if len(source_df) != len(df):
        raise ValueError("unsupervised data row count must match source data row count for PCA visualization")

    sample_idx = df.sample(min(sample_size, len(df)), random_state=42).index
    features, cat_cols, num_cols = select_features(source_df, target)
    preprocessor = make_preprocessor(cat_cols, num_cols)
    x = preprocessor.fit_transform(source_df.loc[sample_idx, features])
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    embedding = PCA(n_components=2, random_state=42).fit_transform(x)

    paths = []
    plot_df = df.loc[sample_idx].copy()
    plot_df["pca_1"] = embedding[:, 0]
    plot_df["pca_2"] = embedding[:, 1]

    color_cols = [col for col in ["cluster_id", target] + score_cols if col in plot_df.columns]
    for col in color_cols:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        if col == "cluster_id":
            codes = plot_df[col].astype("category").cat.codes
            scatter = ax.scatter(plot_df["pca_1"], plot_df["pca_2"], c=codes, s=5, alpha=0.55, cmap="tab20")
        else:
            scatter = ax.scatter(plot_df["pca_1"], plot_df["pca_2"], c=plot_df[col], s=5, alpha=0.55, cmap="viridis")
            fig.colorbar(scatter, ax=ax, label=col)
        ax.set_title(f"PCA embedding colored by {col}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        fig.tight_layout()
        path = output_dir / f"pca_{col}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize unsupervised anomaly and clustering scores.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--unsupervised-data", default=str(UNSUPERVISED_DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-columns", nargs="+", default=DEFAULT_SCORE_COLUMNS)
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--hist-clip-quantile", type=float, default=0.999)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.unsupervised_data, low_memory=False)
    score_cols = [col for col in args.score_columns if col in df.columns]
    if not score_cols:
        raise ValueError("no requested score columns were found in the unsupervised dataset")

    _, valid_mask, test_mask, split_report = make_splits(load_training_data(Path(args.data), args.target))
    df["split"] = "train"
    df.loc[valid_mask.to_numpy(), "split"] = "valid"
    df.loc[test_mask.to_numpy(), "split"] = "test"

    plot_paths = []
    plot_paths.extend(save_histograms(df, score_cols, output_dir, args.hist_clip_quantile))
    plot_paths.extend(save_target_scatter(df, score_cols, args.target, output_dir))
    plot_paths.extend(save_pca_plots(df, Path(args.data), args.target, score_cols, output_dir, args.sample_size))

    report = {
        "score_columns": score_cols,
        "split": split_report,
        "plots": plot_paths,
    }
    report_path = output_dir / "visualization_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
