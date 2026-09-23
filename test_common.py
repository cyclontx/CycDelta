"""Shared standalone test runner for CycDelta."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from GNN import QJGNN
from data_pipeline import (
    CycPeptFeatureStore,
    PairedCycPeptDataset,
    SourceParentRegistry,
    TestDataModule,
    default_data_root,
    fit_monomer_property_normalizer,
    read_ids,
)
from external_holdout import ExternalHoldoutFeatureStore, ExternalPairedHoldoutDataset

warnings.filterwarnings("ignore")


TEST_NAMES = ("internal_test", "faris_data", "merz_data", "nielsen_data")


def load_released_weights(model, ckpt_path: str | Path) -> None:
    """Load a checkpoint that stores only ``model.*`` weights."""
    path = Path(ckpt_path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        str(key).removeprefix("model."): value
        for key, value in state.items()
        if str(key).startswith("model.")
    }
    model.model.load_state_dict(model_state or state, strict=True)


def parse_test_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ckpt_path", default="./checkpoints/best.ckpt")
    parser.add_argument("-d_emb", default=128, type=int)
    parser.add_argument("-n_heads", default=4, type=int)
    parser.add_argument("-drop_out", default=0.25, type=float)
    parser.add_argument("-num_gnn_layer", default=2, type=int)
    parser.add_argument("-lr", default=1e-4, type=float)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-gpu_num", default=1, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-data_root", default=default_data_root())
    parser.add_argument(
        "-accelerator",
        default="gpu",
        choices=("gpu", "cpu", "auto"),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for prediction CSV files when prediction export is enabled.",
    )
    return parser.parse_args()


def cache_build_sample(store) -> None:
    """Keep parsed tensors in memory so later seeds do not reread .pt files."""
    original = store.build_sample
    memo: dict[int, dict] = {}

    def wrapped(index: int) -> dict:
        index = int(index)
        if index not in memo:
            memo[index] = original(index)
        return memo[index]

    store.build_sample = wrapped


class PreparedTest:
    """One feature load, with parent registries that can be redrawn per seed."""

    def __init__(
        self,
        datamodule: TestDataModule,
        y_by_dataset: list[dict[int, float]],
        internal_registry: SourceParentRegistry,
        faris_registry: SourceParentRegistry,
        external_datasets: list[ExternalPairedHoldoutDataset],
    ):
        self.datamodule = datamodule
        self.y_by_dataset = y_by_dataset
        self.internal_registry = internal_registry
        self.faris_registry = faris_registry
        self.external_datasets = external_datasets

    def resample(self, seed: int) -> None:
        self.internal_registry.resample(np.random.default_rng(seed))
        if self.faris_registry is not self.internal_registry:
            self.faris_registry.resample(np.random.default_rng(seed))
        for dataset in self.external_datasets:
            dataset.resample(np.random.default_rng(seed))


def prepare_test(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    single_internal_parent: bool,
) -> PreparedTest:
    data_root = Path(data_root).expanduser().resolve()
    split_root = data_root / "data_split" / "holdout_split"
    train_ids = read_ids(split_root / "OD_train.txt")
    valid_ids = read_ids(split_root / "OD_valid.txt")
    test_ids = read_ids(split_root / "OD_test.txt")
    faris_ids = read_ids(split_root / "faris_data.txt")
    all_ids = sorted(set(train_ids) | set(valid_ids) | set(test_ids) | set(faris_ids))
    descriptor_root = data_root / "monomer_physicochemical"
    descriptor_mean, descriptor_std = fit_monomer_property_normalizer(
        descriptor_root, train_ids
    )
    store = CycPeptFeatureStore(
        csv_path=data_root / "CycPeptMPDB" / "cycpept_mixed_converted.csv",
        ids=all_ids,
        unimol_dir=data_root / "unimol" / "residue_stack",
        glycine_path=data_root / "unimol" / "monomer_cls" / "52_our_process.pt",
        descriptor_dir=descriptor_root,
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
    )
    cache_build_sample(store)
    if single_internal_parent:
        internal_sources = {index: "__internal_test__" for index in test_ids}
        internal_registry = SourceParentRegistry(
            internal_sources,
            name="single_internal_parent",
        )
        internal_dataset = PairedCycPeptDataset(
            store,
            internal_registry,
            test_ids,
            name="internal_test",
            source_by_id=internal_sources,
        )
        faris_registry = SourceParentRegistry(
            store.source_by_id, name="faris_parent_registry"
        )
    else:
        internal_registry = SourceParentRegistry(
            store.source_by_id, name="test_parent_registry"
        )
        internal_dataset = PairedCycPeptDataset(
            store, internal_registry, test_ids, name="internal_test"
        )
        faris_registry = internal_registry
    faris_dataset = PairedCycPeptDataset(
        store, faris_registry, faris_ids, name="faris_data"
    )
    external_datasets = []
    y_by_dataset = [store.y_by_id, store.y_by_id]
    for holdout_name, filename, assay in (
        ("merz_data", "merz_data.xlsx", "PAMPA"),
        ("nielsen_data", "Nielsen_data.xlsx", "unknown"),
    ):
        holdout_store = ExternalHoldoutFeatureStore(
            str(split_root / filename),
            holdout_name,
            str(data_root),
            descriptor_mean,
            descriptor_std,
            assay=assay,
        )
        cache_build_sample(holdout_store)
        dataset = ExternalPairedHoldoutDataset(holdout_store, name=holdout_name)
        external_datasets.append(dataset)
        y_by_dataset.append(holdout_store.y_by_id)
    datasets = [internal_dataset, faris_dataset, *external_datasets]
    return PreparedTest(
        TestDataModule(datasets, batch_size, num_workers),
        y_by_dataset,
        internal_registry,
        faris_registry,
        external_datasets,
    )


def build_test_data(
    args: argparse.Namespace,
    single_internal_parent: bool,
) -> tuple[TestDataModule, list[dict[int, float]]]:
    prepared = prepare_test(
        args.data_root,
        args.batch_size,
        args.num_workers,
        single_internal_parent,
    )
    prepared.resample(int(args.seed))
    return prepared.datamodule, prepared.y_by_dataset


class PredictionWriter(pl.Callback):
    def __init__(
        self,
        names: tuple[str, ...],
        y_by_dataset: list[dict[int, float]],
        output_dir: str | Path,
    ):
        self.names = names
        self.y_by_dataset = y_by_dataset
        self.output_dir = Path(output_dir)
        self.rows = {index: [] for index in range(len(names))}

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        for index, parent, prediction, label in zip(
            outputs["index"],
            outputs["parent_index"],
            outputs["prediction"],
            outputs["label"],
        ):
            child_id = int(index)
            parent_id = int(parent)
            parent_value = self.y_by_dataset[dataloader_idx][parent_id]
            self.rows[dataloader_idx].append(
                {
                    "index": child_id,
                    "parent_index": parent_id,
                    "prediction_delta": float(prediction),
                    "label_delta": float(label),
                    "prediction_abs": parent_value + float(prediction),
                    "label_abs": self.y_by_dataset[dataloader_idx][child_id],
                }
            )

    def on_test_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(self.names):
            path = self.output_dir / f"{name}_predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "index",
                        "parent_index",
                        "prediction_delta",
                        "label_delta",
                        "prediction_abs",
                        "label_abs",
                    ),
                )
                writer.writeheader()
                writer.writerows(self.rows[index])


def run_test(
    args: argparse.Namespace,
    single_internal_parent: bool,
    export_predictions: bool,
) -> None:
    datamodule, y_by_dataset = build_test_data(args, single_internal_parent)
    callbacks = (
        [PredictionWriter(TEST_NAMES, y_by_dataset, args.output_dir)]
        if export_predictions
        else []
    )
    model = QJGNN(
        d_emb=args.d_emb,
        n_heads=args.n_heads,
        dropout=args.drop_out,
        num_gnn_layer=args.num_gnn_layer,
        lr=args.lr,
        test_names=TEST_NAMES,
    )
    load_released_weights(model, args.ckpt_path)
    trainer = pl.Trainer(
        devices=args.gpu_num if args.accelerator == "gpu" else 1,
        accelerator=args.accelerator,
        precision=32,
        logger=pl.loggers.CSVLogger(".", name="test_log"),
        callbacks=callbacks,
    )
    trainer.test(model, datamodule=datamodule)
