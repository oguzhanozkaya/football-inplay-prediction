import numpy as np
import pandas as pd
import torch

import fip.train


def test_training_config_reads_architecture_environment(monkeypatch) -> None:
    monkeypatch.setenv("FIP_TEXT_KERNEL_SIZES", "2,4")
    monkeypatch.setenv("FIP_FUSION_HIDDEN_SIZE", "96")
    monkeypatch.setenv("FIP_GRU_HIDDEN_SIZE", "48")
    monkeypatch.setenv("FIP_DATALOADER_WORKERS", "2")
    monkeypatch.setenv("FIP_CACHE_TENSORS_ON_DEVICE", "false")
    monkeypatch.setenv("FIP_MIXED_PRECISION", "false")
    monkeypatch.setenv("FIP_COMPILE_MODEL", "true")

    config = fip.train.TrainingConfig.from_environment()

    assert config.text_kernel_sizes == (2, 4)
    assert config.fusion_hidden_size == 96
    assert config.gru_hidden_size == 48
    assert config.dataloader_workers == 2
    assert not config.cache_tensors_on_device
    assert not config.mixed_precision
    assert config.compile_model


def test_match_tensor_dataset_shapes() -> None:
    dataset = fip.train.MatchTensorDataset(
        np.zeros((2, 9, 4), dtype=float),
        np.zeros((2, 9, 8), dtype=int),
        np.array([0, 2]),
    )

    item = dataset[0]
    assert item["numeric"].shape == (9, 4)
    assert item["tokens"].shape == (9, 8)
    assert item["target"].dtype == torch.long


def test_sequence_array_handles_nested_numpy_cells() -> None:
    series = pd.Series(
        [
            np.array([np.array([1, 2]), np.array([3, 4])], dtype=object),
            np.array([np.array([5, 6]), np.array([7, 8])], dtype=object),
        ]
    )

    array = fip.train._sequence_array(series, int)

    assert array.shape == (2, 2, 2)
    assert array[1, 1, 1] == 8


def test_fusion_gru_classifier_forward_shape() -> None:
    config = fip.train.TrainingConfig(
        text_embedding_dim=8,
        text_channel_count=4,
        text_kernel_sizes=(2, 3),
        numeric_projection_size=6,
        fusion_hidden_size=10,
        gru_hidden_size=12,
    )
    model = fip.train.FusionGRUClassifier(numeric_input_size=5, vocabulary_size=20, config=config)

    logits = model(torch.zeros((3, 9, 5)), torch.zeros((3, 9, 8), dtype=torch.long))

    assert logits.shape == (3, 3)
