import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.modeling import DATA_PATH, ROOT, TARGET, load_training_data, make_splits, metrics, save_predictions, select_features
from utils.scoreboard import metric_row, update_score_csv

RESULT_DIR = ROOT / "res" / "dnn"
MISSING_VALUE = "__MISSING__"
UNKNOWN_VALUE = "__UNKNOWN__"
COMBO_SPECS = {
    "gu_property": ["gu_name", "property_type"],
    "gu_ym": ["gu_name", "ym"],
    "property_ym": ["property_type", "ym"],
    "housing_type_ym": ["주택유형", "ym"],
    "detail_type_ym": ["유형", "ym"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def add_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for new_col, cols in COMBO_SPECS.items():
        if all(col in out.columns for col in cols):
            values = [out[col].astype("string").fillna(MISSING_VALUE) for col in cols]
            out[new_col] = values[0]
            for value in values[1:]:
                out[new_col] = out[new_col] + "__" + value
    return out


def build_feature_lists(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    features, cat_cols, num_cols = select_features(df, target)
    combo_cols = [col for col in COMBO_SPECS if col in df.columns and col not in cat_cols]
    cat_cols = cat_cols + combo_cols
    features = [col for col in features if col not in combo_cols] + combo_cols
    num_cols = [col for col in features if col not in cat_cols and pd.api.types.is_numeric_dtype(df[col])]
    return features, cat_cols, num_cols


class TabularEncoder:
    def __init__(self, cat_cols: list[str], num_cols: list[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.category_maps: dict[str, dict[str, int]] = {}
        self.num_medians: pd.Series | None = None
        self.num_means: pd.Series | None = None
        self.num_stds: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        for col in self.cat_cols:
            values = df[col].astype("string").fillna(MISSING_VALUE)
            uniques = pd.Index(values.unique()).astype(str).sort_values()
            mapping = {UNKNOWN_VALUE: 0, MISSING_VALUE: 1}
            for value in uniques:
                if value not in mapping:
                    mapping[value] = len(mapping)
            self.category_maps[col] = mapping

        nums = df[self.num_cols].apply(pd.to_numeric, errors="coerce") if self.num_cols else pd.DataFrame(index=df.index)
        self.num_medians = nums.median().fillna(0.0)
        filled = nums.fillna(self.num_medians)
        self.num_means = filled.mean().fillna(0.0)
        self.num_stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_arrays = []
        for col in self.cat_cols:
            mapping = self.category_maps[col]
            values = df[col].astype("string").fillna(MISSING_VALUE).astype(str)
            cat_arrays.append(values.map(mapping).fillna(0).astype("int64").to_numpy())
        if cat_arrays:
            cat_x = np.stack(cat_arrays, axis=1)
        else:
            cat_x = np.zeros((len(df), 0), dtype="int64")

        if self.num_cols:
            nums = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            filled = nums.fillna(self.num_medians)
            scaled = (filled - self.num_means) / self.num_stds
            num_x = scaled.astype("float32").to_numpy()
        else:
            num_x = np.zeros((len(df), 0), dtype="float32")
        return cat_x, num_x

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[col]) for col in self.cat_cols]


class TabularDataset:
    def __init__(self, cat_x: np.ndarray, num_x: np.ndarray, y: np.ndarray | None = None):
        import torch

        self.cat_x = torch.as_tensor(cat_x, dtype=torch.long)
        self.num_x = torch.as_tensor(num_x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.cat_x)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.cat_x[idx], self.num_x[idx]
        return self.cat_x[idx], self.num_x[idx], self.y[idx]


def make_model(cardinalities: list[int], num_features: int, hidden: list[int], dropout: float):
    import torch
    from torch import nn

    class _TabularMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, min(50, max(4, (cardinality + 1) // 2))) for cardinality in cardinalities]
            )
            emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
            layers = []
            in_dim = emb_dim + num_features
            for units in hidden:
                layers.extend([nn.Linear(in_dim, units), nn.BatchNorm1d(units), nn.ReLU(), nn.Dropout(dropout)])
                in_dim = units
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, cat_x, num_x):
            if len(self.embeddings) > 0:
                emb = torch.cat([emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
                x = torch.cat([emb, num_x], dim=1) if num_x.shape[1] else emb
            else:
                x = num_x
            return self.net(x)

    return _TabularMLP()


def choose_device(prefer: str) -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_loader(model, loader, device: str) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                cat_x, num_x, _ = batch
            else:
                cat_x, num_x = batch
            cat_x = cat_x.to(device)
            num_x = num_x.to(device)
            preds.append(model(cat_x, num_x).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def train_model(
    model,
    train_loader,
    valid_loader,
    y_valid: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    min_delta: float,
    lr_patience: int,
    lr_factor: float,
):
    import torch
    from torch import nn

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=1e-6,
    )
    loss_fn = nn.MSELoss()
    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for cat_x, num_x, y in train_loader:
            cat_x = cat_x.to(device)
            num_x = num_x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(cat_x, num_x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        valid_pred = predict_loader(model, valid_loader, device)
        valid_rmse = metrics(y_valid, valid_pred)["rmse"]
        improved = valid_rmse < (best_rmse - min_delta)
        if improved:
            best_rmse = valid_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step(valid_rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} train_loss={np.mean(train_losses):.6f} "
            f"valid_rmse={valid_rmse:.6f} best={best_rmse:.6f} "
            f"stale={stale}/{patience} lr={current_lr:.2e}"
        )
        if epoch >= min_epochs and stale >= patience:
            print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_valid_rmse": best_rmse,
        "min_epochs": min_epochs,
        "min_delta": min_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding MLP DNN for LH market-relative target.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--score-path", default=str(ROOT / "res" / "score.csv"))
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    set_seed(args.random_state)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_df = df.loc[train_mask, features]
    valid_df = df.loc[valid_mask, features]
    test_df = df.loc[test_mask, features]
    y_train = df.loc[train_mask, args.target].to_numpy(dtype="float32")
    y_valid = df.loc[valid_mask, args.target].to_numpy(dtype="float32")
    y_test = df.loc[test_mask, args.target].to_numpy(dtype="float32")

    encoder = TabularEncoder(cat_cols, num_cols).fit(train_df)
    train_cat, train_num = encoder.transform(train_df)
    valid_cat, valid_num = encoder.transform(valid_df)
    test_cat, test_num = encoder.transform(test_df)

    train_loader = DataLoader(TabularDataset(train_cat, train_num, y_train), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TabularDataset(valid_cat, valid_num, y_valid), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(test_cat, test_num, y_test), batch_size=args.batch_size, shuffle=False)

    device = choose_device(args.device)
    print(f"device={device} features={len(features)} cat={len(cat_cols)} num={len(num_cols)}")
    model = make_model(encoder.cardinalities, len(num_cols), args.hidden, args.dropout)
    train_info = train_model(
        model,
        train_loader,
        valid_loader,
        y_valid,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_epochs=args.min_epochs,
        min_delta=args.min_delta,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
    )

    valid_pred = predict_loader(model, valid_loader, device)
    test_pred = predict_loader(model, test_loader, device)
    valid_scores = metrics(y_valid, valid_pred)
    test_scores = metrics(y_test, test_pred)

    report = {
        "target": args.target,
        "split": split_report,
        "features": {"count": len(features), "categorical": cat_cols, "numeric": num_cols},
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden": args.hidden,
            "patience": args.patience,
            "min_epochs": args.min_epochs,
            "min_delta": args.min_delta,
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "device": device,
        },
        "training": train_info,
        "valid": {"models": {"dnn": valid_scores}},
        "test": {"models": {"dnn": test_scores}},
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    valid_pred_path = output_dir / "valid_predictions.csv"
    test_pred_path = output_dir / "test_predictions.csv"
    save_predictions(df, valid_mask, y_valid, {"dnn": valid_pred}, valid_pred_path)
    save_predictions(df, test_mask, y_test, {"dnn": test_pred}, test_pred_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "category_maps": encoder.category_maps,
            "num_medians": None if encoder.num_medians is None else encoder.num_medians.to_dict(),
            "num_means": None if encoder.num_means is None else encoder.num_means.to_dict(),
            "num_stds": None if encoder.num_stds is None else encoder.num_stds.to_dict(),
            "features": features,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "cardinalities": encoder.cardinalities,
            "hidden": args.hidden,
            "dropout": args.dropout,
        },
        output_dir / "model.pt",
    )

    update_score_csv(
        Path(args.score_path),
        [
            metric_row(
                experiment="dnn",
                model="dnn",
                split="test",
                metrics=test_scores,
                target=args.target,
                result_path=metrics_path.relative_to(ROOT),
                rows=split_report["rows"]["test"],
                run_name=f"mlp_{'-'.join(map(str, args.hidden))}_dropout{args.dropout}",
                notes=f"best_epoch={train_info['best_epoch']}",
            )
        ],
    )

    print(f"dnn valid {valid_scores}")
    print(f"dnn test {test_scores}")
    print(f"saved {metrics_path.relative_to(ROOT)}")
    print(f"saved {valid_pred_path.relative_to(ROOT)}")
    print(f"saved {test_pred_path.relative_to(ROOT)}")
    print(f"saved {(output_dir / 'model.pt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
