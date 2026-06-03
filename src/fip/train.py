"""Train the single football text-numeric fusion classifier."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import fip.utils

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _int_tuple_from_environment(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw_value = os.environ.get(name)
    if not raw_value:
        return default
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime training configuration."""

    seed: int = 447
    epochs: int = 80
    patience: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    early_stopping_min_delta: float = 1e-5
    device: str = "cuda"
    text_embedding_dim: int = 64
    text_channel_count: int = 48
    text_kernel_sizes: tuple[int, ...] = (3, 4, 5)
    text_dropout: float = 0.20
    numeric_projection_size: int = 64
    fusion_hidden_size: int = 128
    gru_hidden_size: int = 128
    dropout: float = 0.20

    @classmethod
    def from_environment(cls) -> TrainingConfig:
        return cls(
            seed=int(os.environ.get("FIP_SEED", "447")),
            epochs=int(os.environ.get("FIP_EPOCHS", "80")),
            patience=int(os.environ.get("FIP_PATIENCE", "12")),
            batch_size=int(os.environ.get("FIP_BATCH_SIZE", "64")),
            learning_rate=float(os.environ.get("FIP_LEARNING_RATE", "0.0001")),
            weight_decay=float(os.environ.get("FIP_WEIGHT_DECAY", "0.0")),
            early_stopping_min_delta=float(os.environ.get("FIP_EARLY_STOPPING_MIN_DELTA", "0.00001")),
            device=os.environ.get("FIP_DEVICE", "cuda"),
            text_embedding_dim=int(os.environ.get("FIP_TEXT_EMBEDDING_DIM", "64")),
            text_channel_count=int(os.environ.get("FIP_TEXT_CHANNEL_COUNT", "48")),
            text_kernel_sizes=_int_tuple_from_environment("FIP_TEXT_KERNEL_SIZES", (3, 4, 5)),
            text_dropout=float(os.environ.get("FIP_TEXT_DROPOUT", "0.20")),
            numeric_projection_size=int(os.environ.get("FIP_NUMERIC_PROJECTION_SIZE", "64")),
            fusion_hidden_size=int(os.environ.get("FIP_FUSION_HIDDEN_SIZE", "128")),
            gru_hidden_size=int(os.environ.get("FIP_GRU_HIDDEN_SIZE", "128")),
            dropout=float(os.environ.get("FIP_DROPOUT", "0.20")),
        )


@dataclass(frozen=True)
class TrainingResult:
    predictions_csv_path: Path
    predictions_parquet_path: Path
    summary_path: Path
    history_csv_path: Path
    history_markdown_path: Path
    training_figure_paths: tuple[Path, ...]
    model_path: Path
    prediction_rows: int


class TrainingError(RuntimeError):
    """Raised when processed features are missing or invalid."""


class MatchTensorDataset(Dataset):
    """Tensor dataset for match-window text and numeric sequences."""

    def __init__(self, numeric_sequence: np.ndarray, token_windows: np.ndarray, target: np.ndarray) -> None:
        self.numeric_sequence = torch.tensor(numeric_sequence, dtype=torch.float32)
        self.token_windows = torch.tensor(token_windows, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric_sequence[index],
            "tokens": self.token_windows[index],
            "target": self.target[index],
        }


class WindowTextCNN(nn.Module):
    """CNN text encoder applied independently to each match-time window."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        channel_count: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channel_count, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = channel_count * len(kernel_sizes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, window_count, token_count = token_ids.shape
        flat_tokens = token_ids.reshape(batch_size * window_count, token_count)
        embedded = self.embedding(flat_tokens).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            pooled.append(torch.max(activation, dim=2).values)
        encoded = self.dropout(torch.cat(pooled, dim=1))
        return encoded.reshape(batch_size, window_count, self.output_size)


class FusionGRUClassifier(nn.Module):
    """Single hybrid architecture for minute-45 home/draw/away prediction."""

    def __init__(
        self,
        numeric_input_size: int,
        vocabulary_size: int,
        config: TrainingConfig,
        class_count: int = 3,
    ) -> None:
        super().__init__()
        self.text_encoder = WindowTextCNN(
            vocabulary_size,
            config.text_embedding_dim,
            config.text_channel_count,
            config.text_kernel_sizes,
            config.text_dropout,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_input_size, config.numeric_projection_size),
            nn.ReLU(),
        )
        fused_size = self.text_encoder.output_size + config.numeric_projection_size
        self.fusion_projection = nn.Sequential(
            nn.Linear(fused_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.gru = nn.GRU(config.fusion_hidden_size, config.gru_hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.gru_hidden_size, class_count),
        )

    def forward(self, numeric_sequence: torch.Tensor, token_windows: torch.Tensor) -> torch.Tensor:
        numeric_representation = self.numeric_projection(numeric_sequence)
        text_representation = self.text_encoder(token_windows)
        fused = self.fusion_projection(torch.cat([numeric_representation, text_representation], dim=2))
        _, hidden = self.gru(fused)
        return self.head(hidden[-1])


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _sequence_array(series: pd.Series, dtype: type) -> np.ndarray:
    """Convert parquet nested-list cells into a dense 3D numpy array."""

    return np.stack([np.stack([np.asarray(window, dtype=dtype) for window in sequence]) for sequence in series])


def _scale_numeric_sequences(dataset: pd.DataFrame) -> np.ndarray:
    sequences = _sequence_array(dataset["numeric_sequence"], float)
    train_mask = dataset["split"].to_numpy() == "train"
    scaler = StandardScaler()
    shape = sequences.shape
    flat = sequences.reshape(shape[0] * shape[1], shape[2])
    train_flat = sequences[train_mask].reshape(train_mask.sum() * shape[1], shape[2])
    scaler.fit(train_flat)
    return scaler.transform(flat).reshape(shape)


def _loader(
    indices: np.ndarray, tensor_dataset: MatchTensorDataset, config: TrainingConfig, *, shuffle: bool
) -> DataLoader:
    return DataLoader(Subset(tensor_dataset, indices.tolist()), batch_size=config.batch_size, shuffle=shuffle)


def _run_epoch(
    model: FusionGRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        numeric = batch["numeric"].to(device)
        tokens = batch["tokens"].to(device)
        target = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(numeric, tokens)
        loss = criterion(logits, target)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(target)
        correct += int((logits.argmax(dim=1) == target).sum().item())
        total += int(len(target))
    return total_loss / max(total, 1), correct / max(total, 1)


def _predict(
    model: FusionGRUClassifier,
    tensor_dataset: MatchTensorDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(Subset(tensor_dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
    model.eval()
    probabilities = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["numeric"].to(device), batch["tokens"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(probs.argmax(axis=1))
    return np.concatenate(probabilities), np.concatenate(predictions)


def _write_history_artifacts(
    paths: fip.utils.ProjectPaths, history: list[dict[str, float | int]]
) -> tuple[Path, Path, tuple[Path, ...]]:
    history_frame = pd.DataFrame(history)
    history_csv = paths.reports / "training_history.csv"
    history_md = paths.reports / "training_history.md"
    history_frame.to_csv(history_csv, index=False)
    lines = [
        "# Training History",
        "",
        "| Epoch | Train Loss | Validation Loss | Train Accuracy | Validation Accuracy |",
        "| ----- | ---------- | --------------- | -------------- | ------------------- |",
    ]
    for row in history_frame.itertuples(index=False):
        lines.append(
            f"| {row.epoch} | {row.train_loss:.6f} | {row.validation_loss:.6f} | "
            f"{row.train_accuracy:.4f} | {row.validation_accuracy:.4f} |"
        )
    history_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure_paths = []
    if not history_frame.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
        plt.plot(history_frame["epoch"], history_frame["validation_loss"], label="Validation")
        plt.title("Fusion GRU Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy")
        plt.legend()
        path = paths.figures / "training_loss_fusion_gru.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(path)
    return history_csv, history_md, tuple(figure_paths)


def _log_training_startup(
    *,
    config: TrainingConfig,
    dataset: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    numeric_sequence: np.ndarray,
    token_windows: np.ndarray,
    metadata: dict[str, object],
    vocabulary_size: int,
    device: torch.device,
) -> None:
    """Print a readable training startup summary."""

    split_summary = ", ".join(f"{split}={len(indices):,}" for split, indices in split_indices.items())
    print("train: startup")
    print(f"train:   data rows={len(dataset):,} splits=[{split_summary}]")
    print(
        "train:   tensors "
        f"numeric_sequence={numeric_sequence.shape} token_windows={token_windows.shape} "
        f"numeric_features={len(metadata['numeric_feature_columns']):,} vocabulary={vocabulary_size:,}"
    )
    print(
        "train:   model "
        f"text_embedding={config.text_embedding_dim} text_channels={config.text_channel_count} "
        f"kernels={config.text_kernel_sizes} numeric_projection={config.numeric_projection_size} "
        f"fusion_hidden={config.fusion_hidden_size} gru_hidden={config.gru_hidden_size} dropout={config.dropout}"
    )
    print(
        "train:   optimization "
        f"epochs={config.epochs} patience={config.patience} batch_size={config.batch_size} "
        f"learning_rate={config.learning_rate:g} weight_decay={config.weight_decay:g} "
        f"min_delta={config.early_stopping_min_delta:g} seed={config.seed} device={device}"
    )


def train_model(
    paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS, config: TrainingConfig | None = None
) -> TrainingResult:
    config = TrainingConfig.from_environment() if config is None else config
    fip.utils.ensure_generated_directories(paths)
    set_random_seed(config.seed)
    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    vocabulary_path = paths.processed_data / "text_vocabulary.json"
    missing = [path for path in (dataset_path, metadata_path, vocabulary_path) if not path.is_file()]
    if missing:
        raise TrainingError("Missing processed artifacts. Run `just preprocess` first.")

    dataset = pd.read_parquet(dataset_path).sort_values(["date", "eventId"]).reset_index(drop=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    numeric_sequence = _scale_numeric_sequences(dataset)
    token_windows = _sequence_array(dataset["token_windows"], int)
    target = dataset["target"].to_numpy(dtype=int)
    tensor_dataset = MatchTensorDataset(numeric_sequence, token_windows, target)
    split_indices = {
        split: np.flatnonzero(dataset["split"].to_numpy() == split) for split in ("train", "validation", "test")
    }
    if len(split_indices["train"]) == 0 or len(split_indices["validation"]) == 0:
        raise TrainingError("Train and validation splits must be non-empty")
    device = select_device(config.device)
    _log_training_startup(
        config=config,
        dataset=dataset,
        split_indices=split_indices,
        numeric_sequence=numeric_sequence,
        token_windows=token_windows,
        metadata=metadata,
        vocabulary_size=len(vocabulary),
        device=device,
    )
    model = FusionGRUClassifier(len(metadata["numeric_feature_columns"]), len(vocabulary), config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _loader(split_indices["train"], tensor_dataset, config, shuffle=True)
    validation_loader = _loader(split_indices["validation"], tensor_dataset, config, shuffle=False)

    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            validation_loss, validation_accuracy = _run_epoch(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"train: epoch={epoch}/{config.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.4f}"
        )
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    rows = []
    for split, indices in split_indices.items():
        if len(indices) == 0:
            continue
        probabilities, predictions = _predict(model, tensor_dataset, indices, device, config.batch_size)
        split_frame = dataset.iloc[indices].reset_index(drop=True)
        for row, probs, prediction in zip(split_frame.itertuples(index=False), probabilities, predictions, strict=True):
            rows.append(
                {
                    "eventId": int(row.eventId),
                    "date": row.date,
                    "league": row.league,
                    "split": split,
                    "actual": int(row.target),
                    "actual_label": row.target_label,
                    "prediction": int(prediction),
                    "prediction_label": fip.utils.ID_TO_LABEL[int(prediction)],
                    "prob_home": float(probs[0]),
                    "prob_draw": float(probs[1]),
                    "prob_away": float(probs[2]),
                    "confidence": float(probs.max()),
                }
            )
    predictions = pd.DataFrame(rows)
    predictions_csv = paths.predictions / "predictions.csv"
    predictions_parquet = paths.predictions / "predictions.parquet"
    predictions.to_csv(predictions_csv, index=False)
    predictions.to_parquet(predictions_parquet, index=False)
    model_path = paths.models / "fusion_gru.pt"
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "metadata": metadata}, model_path)
    summary_path = paths.reports / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_name": "fusion_gru",
                "model_type": "text_numeric_sequence_classifier",
                "config": asdict(config),
                "best_validation_loss": best_validation_loss,
                "epochs_ran": len(history),
                "prediction_rows": len(predictions),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    history_csv, history_md, figures = _write_history_artifacts(paths, history)
    print(f"train: wrote {len(predictions)} prediction rows to {predictions_csv.relative_to(paths.root)}")
    return TrainingResult(
        predictions_csv,
        predictions_parquet,
        summary_path,
        history_csv,
        history_md,
        figures,
        model_path,
        len(predictions),
    )


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
