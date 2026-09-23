#!/usr/bin/env python
"""Train CycDelta with ten-seed validation model selection."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from GNN import QJGNN
from data_pipeline import (
    CycPeptFeatureStore,
    PairedCycPeptDataset,
    SourceParentRegistry,
    TrainingDataModule,
    default_data_root,
    fit_monomer_property_normalizer,
    read_ids,
)

warnings.filterwarnings("ignore")


VALIDATION_SEEDS = tuple(range(10))


def rewrite_checkpoint_as_model_weights(filepath: str | Path) -> None:
    """Keep only CycGNN tensors in the released checkpoint format."""
    try:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key: tensor.detach().cpu()
        for key, tensor in state.items()
        if str(key).startswith("model.")
    }
    if not model_state:
        raise RuntimeError(f"No model.* weights found in {filepath}")
    torch.save({"state_dict": model_state}, filepath)


class ModelWeightsCheckpoint(ModelCheckpoint):
    """Save `checkpoints/best.ckpt` as `{"state_dict": model weights}` only."""

    def __init__(self, *args, **kwargs):
        kwargs["save_weights_only"] = True
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        rewrite_checkpoint_as_model_weights(filepath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CycDelta model.")
    parser.add_argument("-d_emb", default=128, type=int)
    parser.add_argument("-n_heads", default=4, type=int)
    parser.add_argument("-drop_out", default=0.25, type=float)
    parser.add_argument("-num_gnn_layer", default=2, type=int)
    parser.add_argument("-lr", default=1e-4, type=float)
    parser.add_argument(
        "-batch_size",
        default=32,
        type=int,
        help="Final pair count after bidirectional augmentation; must be even.",
    )
    parser.add_argument("-gpu_num", default=1, type=int)
    parser.add_argument("-patience", default=500, type=int)
    parser.add_argument("-max_epochs", default=2000, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument(
        "-seed",
        default=0,
        type=int,
        help="Seed for epoch-wise parent sampling only.",
    )
    parser.add_argument("-data_root", default=default_data_root())
    parser.add_argument(
        "-accelerator",
        default="gpu",
        choices=("gpu", "cpu", "auto"),
        help="Lightning accelerator. Use cpu for a local smoke test.",
    )
    return parser.parse_args()


def build_training_data(args: argparse.Namespace) -> TrainingDataModule:
    data_root = Path(args.data_root).expanduser().resolve()
    split_root = data_root / "data_split" / "holdout_split"
    train_ids = read_ids(split_root / "OD_train.txt")
    valid_ids = read_ids(split_root / "OD_valid.txt")
    internal_test_ids = read_ids(split_root / "OD_test.txt")
    faris_ids = read_ids(split_root / "faris_data.txt")
    all_ids = sorted(
        set(train_ids) | set(valid_ids) | set(internal_test_ids) | set(faris_ids)
    )
    descriptor_root = data_root / "monomer_physicochemical"
    descriptor_mean, descriptor_std = fit_monomer_property_normalizer(
        descriptor_root,
        train_ids,
    )
    store = CycPeptFeatureStore(
        csv_path=data_root / "CycPeptMPDB" / "cycpept_mixed_converted.csv",
        ids=all_ids,
        unimol_dir=data_root / "unimol" / "residue_stack",
        glycine_path=(
            data_root / "unimol" / "monomer_cls" / "52_our_process.pt"
        ),
        descriptor_dir=descriptor_root,
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
    )

    train_source_by_id = {
        index: store.source_by_id[index] for index in train_ids
    }
    train_registry = SourceParentRegistry(
        train_source_by_id,
        name="train_parent_registry",
    )
    train_dataset = PairedCycPeptDataset(
        store,
        train_registry,
        train_ids,
        name="train",
    )
    validation_datasets = []
    for seed in VALIDATION_SEEDS:
        registry = SourceParentRegistry(
            store.source_by_id,
            name=f"validation_seed{seed}",
        )
        registry.resample(np.random.default_rng(seed))
        validation_datasets.append(
            PairedCycPeptDataset(
                store,
                registry,
                valid_ids,
                name=f"valid_seed{seed}",
            )
        )
    return TrainingDataModule(
        train_registry=train_registry,
        train_dataset=train_dataset,
        validation_datasets=validation_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    datamodule = build_training_data(args)
    model = QJGNN(
        d_emb=args.d_emb,
        n_heads=args.n_heads,
        dropout=args.drop_out,
        num_gnn_layer=args.num_gnn_layer,
        lr=args.lr,
        validation_seeds=VALIDATION_SEEDS,
    )
    monitor = "val/pearson_mean"
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            mode="max",
            verbose=True,
        ),
        ModelWeightsCheckpoint(
            dirpath="checkpoints",
            filename="best",
            monitor=monitor,
            mode="max",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
            verbose=True,
        ),
    ]
    trainer = pl.Trainer(
        devices=args.gpu_num if args.accelerator == "gpu" else 1,
        accelerator=args.accelerator,
        precision=32,
        max_epochs=args.max_epochs,
        logger=CSVLogger(".", name="log"),
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1000,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
