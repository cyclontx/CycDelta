"""Shared feature loading, pairing, collation, and data modules for CycDelta."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from external_holdout import ExternalPairedHoldoutDataset


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {amino_acid: index for index, amino_acid in enumerate(AA_VOCAB)}
UNK_IDX = len(AA_VOCAB)
METHOD_VOCAB = ("PAMPA", "CACO2", "MDCK", "RRCK")
METHOD_TO_IDX = {method: index for index, method in enumerate(METHOD_VOCAB)}
GLYCINE_MONOMER_ID = 52
FEATURE_SUFFIX = "_our_process.pt"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def read_ids(path: str | Path) -> list[int]:
    return [
        int(value.strip())
        for value in Path(path).read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def fit_monomer_property_normalizer(
    feature_dir: str | Path,
    train_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_dir = Path(feature_dir)
    matrices = []
    missing = []
    for index in train_ids:
        path = feature_dir / f"{index}{FEATURE_SUFFIX}"
        if not path.exists():
            missing.append(path)
            continue
        matrix = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).float()
        if matrix.ndim != 2 or matrix.size(1) != 31:
            raise ValueError(
                f"Index={index}: monomer descriptor shape is {tuple(matrix.shape)}; "
                "expected [num_residues, 31]"
            )
        matrices.append(matrix)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} descriptor files. Run "
            f"`python prepare_features.py`; first missing file: {missing[0]}"
        )
    values = torch.cat(matrices, dim=0)
    values = values.masked_fill(~torch.isfinite(values), float("nan"))
    mean = torch.nanmean(values, dim=0)
    if not torch.isfinite(mean).all():
        raise ValueError("At least one descriptor has no finite training values")
    finite = torch.isfinite(values)
    centered = torch.where(finite, values - mean, torch.zeros_like(values))
    counts = finite.sum(dim=0).clamp_min(1)
    std = torch.sqrt(centered.square().sum(dim=0) / counts)
    std = torch.where(
        torch.isfinite(std) & (std >= 1e-6),
        std,
        torch.ones_like(std),
    )
    return mean, std


def build_complete_residue_graph(
    num_residues: int,
    cyclized_pair: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    structural_edges = set()
    for index in range(num_residues - 1):
        structural_edges.update(((index, index + 1), (index + 1, index)))
    first, second = cyclized_pair
    structural_edges.update(((first, second), (second, first)))
    edges = []
    attributes = []
    for source in range(num_residues):
        for target in range(num_residues):
            if source == target:
                continue
            edges.append((source, target))
            attributes.append((float((source, target) in structural_edges),))
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attributes, dtype=torch.float32),
    )


class CycPeptFeatureStore:
    """Load only features consumed by the current CycDelta model."""

    def __init__(
        self,
        csv_path: str | Path,
        ids: list[int],
        unimol_dir: str | Path,
        glycine_path: str | Path,
        descriptor_dir: str | Path,
        descriptor_mean: torch.Tensor,
        descriptor_std: torch.Tensor,
    ):
        csv_path = Path(csv_path)
        frame = pd.read_csv(csv_path)
        required = {"Index", "SMILES", "Method", "Source", "Permeability_Value"}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_columns)}"
            )
        frame["Index"] = frame["Index"].astype(int)
        requested = set(ids)
        frame = frame[frame["Index"].isin(requested)].copy()
        missing_ids = requested - set(frame["Index"])
        if missing_ids:
            raise ValueError(f"Requested Index values are absent: {sorted(missing_ids)}")
        methods = frame["Method"].astype(str).str.strip().str.upper()
        unsupported = sorted(set(methods) - set(METHOD_VOCAB))
        if unsupported:
            raise ValueError(f"Unsupported Method values: {unsupported}")
        order = {index: position for position, index in enumerate(ids)}
        frame["_order"] = frame["Index"].map(order)
        frame = frame.sort_values("_order")
        self.row_by_id = {
            int(row["Index"]): row for _, row in frame.iterrows()
        }
        self.source_by_id = {
            index: str(row["Source"]) for index, row in self.row_by_id.items()
        }
        self.y_by_id = {
            index: float(row["Permeability_Value"])
            for index, row in self.row_by_id.items()
        }
        manifest_path = csv_path.with_name("our_process_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.record_by_id = {
            int(record["index"]): record for record in manifest["records"]
        }
        missing_records = requested - set(self.record_by_id)
        if missing_records:
            raise ValueError(
                f"Manifest records are missing for Index values: {sorted(missing_records)}"
            )
        self.unimol_dir = Path(unimol_dir)
        self.descriptor_dir = Path(descriptor_dir)
        self.glycine = torch.load(
            glycine_path,
            map_location="cpu",
            weights_only=True,
        ).float()
        self.descriptor_mean = descriptor_mean
        self.descriptor_std = descriptor_std

    def build_sample(self, index: int) -> dict:
        row = self.row_by_id[index]
        record = self.record_by_id[index]
        residues = record["residues"]
        count = len(residues)
        residue_flags = torch.tensor(
            [
                (
                    float(residue["is_natural"]),
                    float(residue["is_D"]),
                    float(residue["is_methylated"]),
                )
                for residue in residues
            ],
            dtype=torch.float32,
        )
        sequence = torch.zeros((count, 21), dtype=torch.float32)
        for position, amino_acid in enumerate(record["sequence"]):
            sequence[position, AA_TO_IDX.get(str(amino_acid).upper(), UNK_IDX)] = 1.0
        method = str(row["Method"]).strip().upper()
        method_feature = torch.zeros((count, 4), dtype=torch.float32)
        method_feature[:, METHOD_TO_IDX[method]] = 1.0
        node_feature = torch.cat(
            (residue_flags, sequence, method_feature),
            dim=-1,
        )
        unimol = torch.load(
            self.unimol_dir / f"{index}{FEATURE_SUFFIX}",
            map_location="cpu",
            weights_only=True,
        ).float()
        descriptors = torch.load(
            self.descriptor_dir / f"{index}{FEATURE_SUFFIX}",
            map_location="cpu",
            weights_only=True,
        ).float()
        expected = {"unimol": (count, 512), "descriptors": (count, 31)}
        actual = {
            "unimol": tuple(unimol.shape),
            "descriptors": tuple(descriptors.shape),
        }
        if actual != expected:
            raise ValueError(
                f"Index={index}: feature shape mismatch; actual={actual}, "
                f"expected={expected}"
            )
        descriptors = torch.where(
            torch.isfinite(descriptors),
            descriptors,
            self.descriptor_mean,
        )
        descriptors = (
            descriptors - self.descriptor_mean
        ) / self.descriptor_std
        edge_index, edge_attr = build_complete_residue_graph(
            count, record["cyclized_pair"]
        )
        return {
            "index": index,
            "node_feature": node_feature,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "delta_unimol_feature": unimol - self.glycine.unsqueeze(0),
            "monomer_property_feature": descriptors,
            "y": torch.tensor([self.y_by_id[index]], dtype=torch.float32),
        }


class SourceParentRegistry:
    def __init__(self, source_by_id: dict[int, str], name: str = "registry"):
        self.name = name
        grouped = defaultdict(list)
        for index, source in source_by_id.items():
            grouped[source].append(index)
        self.source_to_ids = {
            source: values for source, values in grouped.items() if len(values) >= 2
        }
        self.source_to_parent: dict[str, int] = {}
        self.resample(np.random.default_rng(0))

    def resample(self, rng: np.random.Generator) -> None:
        self.source_to_parent = {
            source: values[int(rng.integers(len(values)))]
            for source, values in self.source_to_ids.items()
        }

    def parent_of(self, source: str) -> int:
        return self.source_to_parent[source]


class PairedCycPeptDataset(Dataset):
    def __init__(
        self,
        store: CycPeptFeatureStore,
        registry: SourceParentRegistry,
        child_ids: list[int],
        name: str,
        source_by_id: dict[int, str] | None = None,
    ):
        self.store = store
        self.registry = registry
        self.name = name
        self.source_by_id = source_by_id or store.source_by_id
        self.child_ids = [
            index
            for index in child_ids
            if self.source_by_id[index] in registry.source_to_ids
        ]

    def __len__(self) -> int:
        return len(self.child_ids)

    def __getitem__(self, position: int) -> dict:
        child_id = self.child_ids[position]
        source = self.source_by_id[child_id]
        parent_id = self.registry.parent_of(source)
        return {
            "child": self.store.build_sample(child_id),
            "parent": self.store.build_sample(parent_id),
            "index": child_id,
            "parent_index": parent_id,
            "y": torch.tensor(
                [self.store.y_by_id[child_id] - self.store.y_by_id[parent_id]],
                dtype=torch.float32,
            ),
        }


def collate_graph_batch(samples: list[dict]) -> dict:
    node_counts = [sample["node_feature"].size(0) for sample in samples]
    offsets = np.cumsum([0] + node_counts[:-1]).tolist()
    return {
        "index": torch.tensor([sample["index"] for sample in samples]),
        "node_feature": torch.cat([sample["node_feature"] for sample in samples]),
        "edge_index": torch.cat(
            [
                sample["edge_index"] + offset
                for sample, offset in zip(samples, offsets)
            ],
            dim=1,
        ),
        "edge_attr": torch.cat([sample["edge_attr"] for sample in samples]),
        "delta_unimol_feature": torch.cat(
            [sample["delta_unimol_feature"] for sample in samples]
        ),
        "monomer_property_feature": torch.cat(
            [sample["monomer_property_feature"] for sample in samples]
        ),
        "node_batch": torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long),
            torch.tensor(node_counts, dtype=torch.long),
        ),
    }


def collate_pair_batch(samples: list[dict]) -> dict:
    return {
        "child": collate_graph_batch([sample["child"] for sample in samples]),
        "parent": collate_graph_batch([sample["parent"] for sample in samples]),
        "index": torch.tensor([sample["index"] for sample in samples]),
        "parent_index": torch.tensor(
            [sample["parent_index"] for sample in samples]
        ),
        "y": torch.cat([sample["y"] for sample in samples]),
    }


def collate_bidirectional_pair_batch(samples: list[dict]) -> dict:
    augmented = []
    for sample in samples:
        augmented.extend(
            (
                sample,
                {
                    "child": sample["parent"],
                    "parent": sample["child"],
                    "index": sample["parent_index"],
                    "parent_index": sample["index"],
                    "y": -sample["y"],
                },
            )
        )
    return collate_pair_batch(augmented)


class TrainingDataModule(pl.LightningDataModule):
    """Train with epoch resampling and validate over fixed seeds 0 through 9."""

    def __init__(
        self,
        train_registry: SourceParentRegistry,
        train_dataset: PairedCycPeptDataset,
        validation_datasets: list[PairedCycPeptDataset],
        batch_size: int,
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Bidirectional augmentation requires an even batch size")
        self.train_registry = train_registry
        self.train_dataset = train_dataset
        self.validation_datasets = validation_datasets
        self.batch_size = batch_size
        self.base_batch_size = batch_size // 2
        self.num_workers = num_workers
        self.seed = seed
        self._resampled_epoch = None

    def _resample_train(self) -> None:
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        if epoch != self._resampled_epoch:
            self.train_registry.resample(np.random.default_rng(self.seed + epoch))
            self._resampled_epoch = epoch

    def train_dataloader(self) -> DataLoader:
        self._resample_train()
        return DataLoader(
            self.train_dataset,
            batch_size=self.base_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_bidirectional_pair_batch,
            pin_memory=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.base_batch_size,
                shuffle=False,
                num_workers=max(0, self.num_workers // 2),
                collate_fn=collate_bidirectional_pair_batch,
                pin_memory=True,
            )
            for dataset in self.validation_datasets
        ]


class TestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: list[Dataset],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=max(0, self.num_workers // 2),
                collate_fn=collate_pair_batch,
                pin_memory=True,
            )
            for dataset in self.datasets
        ]


def default_data_root() -> str:
    return os.environ.get("PERMEABILITY_DATA_ROOT", str(DEFAULT_DATA_ROOT))
