"""Feature store and paired dataset for XLSX-based external holdouts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from process_monomer_descriptors import descriptor_matrix


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {amino_acid: index for index, amino_acid in enumerate(AA_VOCAB)}
UNK_IDX = len(AA_VOCAB)
GLYCINE_MONOMER_ID = 52


def _read_external_table(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    if "CAPA" in frame.columns and "Permeability_Value" not in frame.columns:
        frame["Permeability_Value"] = frame["CAPA"]
    if "Index" not in frame.columns:
        frame.insert(0, "Index", range(len(frame)))
    required = {"Index", "SMILES", "Permeability_Value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame.loc[:, ["Index", "SMILES", "Permeability_Value"]].copy()


def build_complete_residue_graph(
    num_residues: int,
    cyclized_pair,
) -> tuple[torch.Tensor, torch.Tensor]:
    structural_edges = set()
    for index in range(num_residues - 1):
        structural_edges.add((index, index + 1))
        structural_edges.add((index + 1, index))
    first, second = cyclized_pair
    structural_edges.add((first, second))
    structural_edges.add((second, first))

    edges = []
    attributes = []
    for source in range(num_residues):
        for target in range(num_residues):
            if source == target:
                continue
            edges.append((source, target))
            attributes.append((1.0 if (source, target) in structural_edges else 0.0,))
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attributes, dtype=torch.float32),
    )


class ExternalHoldoutFeatureStore:
    def __init__(
        self,
        xlsx_path: str,
        holdout_name: str,
        data_root: str,
        monomer_property_mean: torch.Tensor | None = None,
        monomer_property_std: torch.Tensor | None = None,
        assay: str = "PAMPA",
    ):
        self.holdout_name = holdout_name
        self.data_root = Path(data_root)
        self.xlsx_path = Path(xlsx_path)
        self.assay = assay
        manifest_path = self.xlsx_path.with_name(
            f"{holdout_name}_our_process_manifest.json"
        )

        dataframe = _read_external_table(self.xlsx_path)
        dataframe["Index"] = dataframe["Index"].astype(int)
        if dataframe["Index"].duplicated().any():
            raise ValueError(f"{self.xlsx_path}: duplicate Index values")
        self.y_by_id = {
            int(row["Index"]): float(row["Permeability_Value"])
            for _, row in dataframe.iterrows()
        }

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.record_by_id = {
            int(record["index"]): record for record in manifest["records"]
        }
        missing = set(self.y_by_id) - set(self.record_by_id)
        if missing:
            raise ValueError(
                f"{holdout_name}: preprocessing failed or is missing for Index "
                f"{sorted(missing)}; see {manifest_path}"
            )
        self.ids = list(self.y_by_id)
        self.unimol_dir = self.data_root / "unimol" / "residue_stack"
        self.glycine_feature = torch.load(
            self.data_root
            / "unimol"
            / "monomer_cls"
            / f"{GLYCINE_MONOMER_ID}_our_process.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        if (monomer_property_mean is None) != (monomer_property_std is None):
            raise ValueError(
                "monomer_property_mean and monomer_property_std must be provided together"
            )
        self.monomer_property_by_id = {}
        if monomer_property_mean is not None:
            if tuple(monomer_property_mean.shape) != (31,) or tuple(
                monomer_property_std.shape
            ) != (31,):
                raise ValueError("Monomer property normalizer must have shape [31]")
            for index, record in self.record_by_id.items():
                features = descriptor_matrix(record["residues"])
                features = torch.where(
                    torch.isfinite(features),
                    features,
                    monomer_property_mean,
                )
                self.monomer_property_by_id[index] = (
                    features - monomer_property_mean
                ) / monomer_property_std

    def build_sample(self, index: int) -> dict:
        record = self.record_by_id[index]
        residues = record["residues"]
        num_residues = len(residues)

        residue_properties = torch.tensor(
            [
                [
                    float(residue["is_natural"]),
                    float(residue["is_D"]),
                    float(residue["is_methylated"]),
                ]
                for residue in residues
            ],
            dtype=torch.float32,
        )
        sequence_onehot = torch.zeros((num_residues, 21), dtype=torch.float32)
        for position, amino_acid in enumerate(record["sequence"]):
            sequence_onehot[position, AA_TO_IDX.get(amino_acid, UNK_IDX)] = 1.0
        method_onehot = torch.zeros((num_residues, 4), dtype=torch.float32)
        if self.assay == "PAMPA":
            method_onehot[:, 0] = 1.0
        elif self.assay != "unknown":
            raise ValueError(f"Unsupported external assay: {self.assay}")
        node_feature = torch.cat(
            [residue_properties, sequence_onehot, method_onehot],
            dim=-1,
        )

        feature_name = f"{self.holdout_name}_{index}.pt"
        unimol = torch.load(
            self.unimol_dir / feature_name,
            map_location="cpu",
            weights_only=True,
        ).float()
        delta_unimol = unimol - self.glycine_feature.unsqueeze(0)
        expected = {
            "unimol": (num_residues, 512),
            "node": (num_residues, 28),
        }
        actual = {
            "unimol": tuple(unimol.shape),
            "node": tuple(node_feature.shape),
        }
        if actual != expected:
            raise ValueError(
                f"{self.holdout_name} Index={index}: feature shape mismatch; "
                f"actual={actual}, expected={expected}"
            )
        monomer_property = self.monomer_property_by_id.get(
            index,
            torch.zeros((num_residues, 31), dtype=torch.float32),
        )

        edge_index, edge_attr = build_complete_residue_graph(
            num_residues,
            record["cyclized_pair"],
        )
        return {
            "index": index,
            "node_feature": node_feature,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "delta_unimol_feature": delta_unimol,
            "monomer_property_feature": monomer_property,
            "y": torch.tensor([self.y_by_id[index]], dtype=torch.float32),
        }


class ExternalPairedHoldoutDataset(Dataset):
    """Treat one XLSX as one Source and compare every row with one random parent."""

    def __init__(self, store: ExternalHoldoutFeatureStore, name: str):
        if len(store.ids) < 2:
            raise ValueError(f"{name} requires at least two rows to form differences")
        self.store = store
        self.name = name
        self.parent_id = store.ids[0]

    def resample(self, rng) -> None:
        self.parent_id = self.store.ids[int(rng.integers(len(self.store.ids)))]

    def __len__(self) -> int:
        return len(self.store.ids)

    def __getitem__(self, position: int) -> dict:
        child_id = self.store.ids[position]
        parent_id = self.parent_id
        delta = self.store.y_by_id[child_id] - self.store.y_by_id[parent_id]
        return {
            "child": self.store.build_sample(child_id),
            "parent": self.store.build_sample(parent_id),
            "index": child_id,
            "parent_index": parent_id,
            "y": torch.tensor([delta], dtype=torch.float32),
        }
