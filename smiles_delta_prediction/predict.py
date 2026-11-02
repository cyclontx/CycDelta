#!/usr/bin/env python
"""Predict zero-corrected delta and absolute permeability from query SMILES."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from GNN import CycGNN  # noqa: E402
from peptide_preprocessing import AA_VOCAB, parse_cyclic_peptide  # noqa: E402
from process_monomer_descriptors import (  # noqa: E402
    DESCRIPTOR_NAMES,
    descriptor_matrix,
)


METHODS = ("PAMPA", "CACO2", "MDCK", "RRCK")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_VOCAB)}
UNK_INDEX = len(AA_VOCAB)
GLYCINE_SMILES = "NCC(=O)O"
OUTPUT_FIELDS = (
    "line_number",
    "smiles",
    "method",
    "reference_source",
    "reference_smiles",
    "delta_log10",
    "absolute_log10_cm_s",
    "absolute_1e-6_cm_s",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict permeability from SMILES. Use the nearest training "
            "reference at Tanimoto >= 0.7; otherwise use the first input."
        )
    )
    parser.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "smiles.txt"),
        help="One query cyclic-peptide SMILES per non-empty line",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "predictions.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            MODEL_DIR
            / "checkpoints"
            / "best.ckpt"
        ),
    )
    parser.add_argument(
        "--normalizer",
        default=str(SCRIPT_DIR / "descriptor_normalizer.json"),
    )
    parser.add_argument(
        "--method",
        choices=(*METHODS, "unknown"),
        default="PAMPA",
    )
    parser.add_argument(
        "--reference-smiles",
        default=None,
        help=(
            "Optional reference SMILES used for all queries. A measured "
            "reference value must also be supplied."
        ),
    )
    parser.add_argument(
        "--reference-value-log10",
        type=float,
        default=None,
        help="Measured reference log10(Papp [cm/s])",
    )
    parser.add_argument(
        "--reference-value-1e-6",
        type=float,
        default=None,
        help="Measured reference Papp in 10^-6 cm/s",
    )
    parser.add_argument(
        "--train-csv",
        default=str(MODEL_DIR / "data" / "CycPeptMPDB" / "cycpept_mixed_converted.csv"),
    )
    parser.add_argument(
        "--train-ids",
        default=str(
            MODEL_DIR / "data" / "data_split" / "holdout_split" / "OD_train.txt"
        ),
        help="Training ID file that limits eligible references",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unimol-batch-size", type=int, default=128)
    parser.add_argument(
        "--remove-hs",
        action="store_true",
        help="Pass remove_hs to Uni-Mol; hydrogens are retained by default",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def read_input(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows = [
        {"line_number": line_number, "smiles": text.strip()}
        for line_number, text in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        )
        if text.strip()
    ]
    if not rows:
        raise ValueError(f"Input file contains no non-empty SMILES: {path}")
    return rows


def parse_records(rows: list[dict[str, Any]]) -> tuple[list[Any], dict[int, str]]:
    records: list[Any] = []
    errors: dict[int, str] = {}
    for row in rows:
        try:
            records.append(
                parse_cyclic_peptide(row["smiles"], int(row["line_number"]))
            )
        except Exception as exc:
            errors[int(row["line_number"])] = str(exc)
            records.append(None)
    return records, errors


def load_normalizer(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(payload["descriptor_names"])
    if names != tuple(DESCRIPTOR_NAMES):
        raise ValueError("Descriptor order in descriptor_normalizer.json does not match")
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    expected = (len(DESCRIPTOR_NAMES),)
    if tuple(mean.shape) != expected or tuple(std.shape) != expected:
        raise ValueError(
            f"Invalid normalizer shapes: mean={tuple(mean.shape)}, "
            f"std={tuple(std.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Normalizer contains non-finite values")
    if not bool((std > 0).all()):
        raise ValueError("All normalizer standard deviations must be positive")
    return mean, std


def normalize_descriptors(
    values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    values = torch.where(torch.isfinite(values), values, mean)
    return (values - mean) / std


def load_model(checkpoint_path: Path, device: torch.device) -> CycGNN:
    model = CycGNN(
        d_emb=128,
        n_heads=4,
        dropout=0.25,
        num_gnn_layer=2,
    )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state or state, strict=True)
    model.eval().to(device)
    return model


def encode_unimol(
    smiles: list[str],
    batch_size: int,
    remove_hs: bool,
) -> torch.Tensor:
    from unimol_tools import UniMolRepr

    model = UniMolRepr(data_type="molecule", remove_hs=remove_hs)
    batches = []
    try:
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            values = np.asarray(
                model.get_repr(batch, return_atomic_reprs=True)["cls_repr"],
                dtype=np.float32,
            )
            if tuple(values.shape) != (len(batch), 512):
                raise ValueError(f"Unexpected Uni-Mol output shape: {values.shape}")
            batches.append(torch.from_numpy(values))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def build_edges(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    edges: list[tuple[int, int]] = []
    attributes: list[tuple[float]] = []
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            edges.append((source, target))
            structural = (
                abs(source - target) == 1
                or {source, target} == {0, count - 1}
            )
            attributes.append((float(structural),))
    if not edges:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attributes, dtype=torch.float32),
    )


def node_features(record: Any, method: str) -> torch.Tensor:
    features = torch.zeros((len(record.residues), 28), dtype=torch.float32)
    for position, residue in enumerate(record.residues):
        features[position, :3] = torch.tensor(
            [residue.is_natural, residue.is_D, residue.is_methylated],
            dtype=torch.float32,
        )
        aa_index = AA_TO_INDEX.get(residue.amino_acid, UNK_INDEX)
        features[position, 3 + aa_index] = 1
        if method != "unknown":
            features[position, 24 + METHODS.index(method)] = 1
    return features


def make_graph(
    record: Any,
    method: str,
    unimol: torch.Tensor,
    glycine: torch.Tensor,
    properties: torch.Tensor,
) -> dict[str, torch.Tensor]:
    count = len(record.residues)
    if tuple(unimol.shape) != (count, 512):
        raise ValueError(
            f"Uni-Mol feature shape does not match residue count: "
            f"{tuple(unimol.shape)}, L={count}"
        )
    edge_index, edge_attr = build_edges(count)
    return {
        "node_feature": node_features(record, method),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "delta_unimol_feature": unimol - glycine.unsqueeze(0),
        "monomer_property_feature": properties,
    }


def collate_graphs(
    graphs: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    node_counts = [graph["node_feature"].size(0) for graph in graphs]
    offsets = np.cumsum([0] + node_counts[:-1]).tolist()
    batch = {
        "node_feature": torch.cat([graph["node_feature"] for graph in graphs]),
        "edge_index": torch.cat(
            [
                graph["edge_index"] + offset
                for graph, offset in zip(graphs, offsets)
            ],
            dim=1,
        ),
        "edge_attr": torch.cat([graph["edge_attr"] for graph in graphs]),
        "delta_unimol_feature": torch.cat(
            [graph["delta_unimol_feature"] for graph in graphs]
        ),
        "monomer_property_feature": torch.cat(
            [graph["monomer_property_feature"] for graph in graphs]
        ),
    }
    batch["node_batch"] = torch.repeat_interleave(
        torch.arange(len(graphs), dtype=torch.long),
        torch.tensor(node_counts, dtype=torch.long),
    )
    return batch


def move_graph(
    graph: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in graph.items()}


@torch.inference_mode()
def predict_deltas(
    model: CycGNN,
    graphs: list[dict[str, torch.Tensor]],
    parent: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    predictions: list[float] = []
    parent_batch = move_graph(collate_graphs([parent]), device)
    parent_embedding = model._pool_graph(parent_batch, 1)
    zero_baseline = model.out_mlp(torch.zeros_like(parent_embedding)).squeeze(-1)
    for start in range(0, len(graphs), batch_size):
        children = graphs[start : start + batch_size]
        child_batch = move_graph(collate_graphs(children), device)
        child_embedding = model._pool_graph(child_batch, len(children))
        differences = child_embedding - parent_embedding.expand(len(children), -1)
        values = model.out_mlp(differences).squeeze(-1) - zero_baseline
        items = values.detach().cpu().tolist()
        for index, child in enumerate(children):
            if child is parent:
                items[index] = 0.0
        predictions.extend(items)
    return predictions


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def load_reference_catalog(
    csv_path: Path,
    ids_path: Path,
    method: str,
) -> list[dict[str, Any]]:
    if method == "unknown":
        raise ValueError(
            "The default training-reference mode does not support method=unknown"
        )
    train_ids = {
        int(value.strip())
        for value in ids_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    frame = pd.read_csv(csv_path)
    required = {"Index", "SMILES", "Method", "Permeability_Value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    frame = frame[
        frame["Index"].astype(int).isin(train_ids)
        & (frame["Method"].astype(str).str.upper() == method)
    ].copy()
    frame["canonical_smiles"] = frame["SMILES"].map(canonical_smiles)
    frame["Permeability_Value"] = pd.to_numeric(
        frame["Permeability_Value"], errors="raise"
    )

    catalog = []
    for canonical, group in frame.groupby("canonical_smiles", sort=False):
        catalog.append(
            {
                "id": "|".join(group["Index"].astype(int).astype(str)),
                "smiles": str(group.iloc[0]["SMILES"]),
                "canonical_smiles": canonical,
                "value_log10": float(group["Permeability_Value"].mean()),
            }
        )
    if not catalog:
        raise ValueError(f"No eligible method={method} references in training data")
    return catalog


def select_most_similar_reference(
    query_smiles: str,
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query = Chem.MolFromSmiles(query_smiles)
    query_fp = generator.GetFingerprint(query)
    reference_fps = [
        generator.GetFingerprint(Chem.MolFromSmiles(item["smiles"]))
        for item in catalog
    ]
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
    index = int(np.argmax(similarities))
    item = dict(catalog[index])
    item["similarity"] = float(similarities[index])
    return item


def resolve_custom_reference(
    smiles: str,
    value_log10: float | None,
    value_1e6: float | None,
) -> dict[str, Any]:
    if value_log10 is not None and value_1e6 is not None:
        raise ValueError(
            "Provide only one of --reference-value-log10 and "
            "--reference-value-1e-6"
        )
    canonical = canonical_smiles(smiles)
    if value_1e6 is not None:
        if value_1e6 <= 0:
            raise ValueError("--reference-value-1e-6 must be positive")
        value_log10 = float(np.log10(value_1e6 * 1e-6))
    if value_log10 is None:
        raise ValueError(
            "A user reference requires --reference-value-log10 or "
            "--reference-value-1e-6"
        )
    return {
        "id": "user",
        "smiles": smiles,
        "canonical_smiles": canonical,
        "value_log10": float(value_log10),
        "similarity": "",
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.unimol_batch_size < 1:
        raise ValueError("All batch sizes must be positive")
    if args.reference_smiles is None and (
        args.reference_value_log10 is not None
        or args.reference_value_1e_6 is not None
    ):
        raise ValueError(
            "A reference value requires --reference-smiles"
        )

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    normalizer_path = Path(args.normalizer).expanduser().resolve()
    train_csv_path = Path(args.train_csv).expanduser().resolve()
    train_ids_path = Path(args.train_ids).expanduser().resolve()
    device = resolve_device(args.device)

    rows = read_input(input_path)
    records, errors = parse_records(rows)
    if records[0] is None:
        raise ValueError(
            "The first SMILES must parse successfully because it is the "
            "fallback reference when no similar training reference is found"
        )
    valid_positions = [
        index for index, record in enumerate(records) if record is not None
    ]
    valid_records = [records[index] for index in valid_positions]
    if not valid_records:
        raise ValueError("All input SMILES failed preprocessing")
    print(
        f"Parsed {len(valid_records)}/{len(rows)} query SMILES",
        flush=True,
    )

    if args.reference_smiles is not None:
        reference = resolve_custom_reference(
            args.reference_smiles,
            args.reference_value_log10,
            args.reference_value_1e_6,
        )
        reference_source = "user"
        nearest_similarity: float | None = None
        message = (
            "Using the user-specified reference and measured permeability. "
            "Relative and absolute permeability values are available."
        )
    else:
        best_training_reference = None
        if args.method != "unknown":
            catalog = load_reference_catalog(
                train_csv_path,
                train_ids_path,
                args.method,
            )
            best_training_reference = select_most_similar_reference(
                rows[0]["smiles"], catalog
            )
        nearest_similarity = (
            None
            if best_training_reference is None
            else float(best_training_reference["similarity"])
        )
        if (
            best_training_reference is not None
            and nearest_similarity >= 0.7
        ):
            reference = best_training_reference
            reference_source = "training_nearest"
            message = (
                f"Using a training reference with Tanimoto similarity "
                f"{nearest_similarity:.3f} >= 0.700. Relative and absolute "
                "permeability values are available."
            )
        else:
            reference = {
                "id": "input_line_1",
                "smiles": rows[0]["smiles"],
                "canonical_smiles": canonical_smiles(rows[0]["smiles"]),
                "value_log10": None,
                "similarity": nearest_similarity,
            }
            reference_source = "input_first_line"
            similarity_text = (
                "unavailable"
                if nearest_similarity is None
                else f"{nearest_similarity:.3f}"
            )
            message = (
                f"WARNING: The highest training similarity was "
                f"{similarity_text}, below 0.700. The first input was used "
                "as the reference, so only relative permeability values are "
                "available. Provide --reference-smiles and its measured "
                "permeability to obtain absolute values."
            )

    reference_record = parse_cyclic_peptide(reference["smiles"], -1)

    mean, std = load_normalizer(normalizer_path)
    feature_records = valid_records + [reference_record]
    residue_counts = [len(record.residues) for record in feature_records]
    residue_smiles = [
        residue.smiles
        for record in feature_records
        for residue in record.residues
    ]
    print(f"Calculating Uni-Mol for {len(residue_smiles)} residues...", flush=True)
    all_unimol = encode_unimol(
        residue_smiles + [GLYCINE_SMILES],
        args.unimol_batch_size,
        args.remove_hs,
    )
    glycine = all_unimol[-1]
    all_unimol = all_unimol[:-1]
    unimol_by_record = list(torch.split(all_unimol, residue_counts))

    print("Calculating RDKit descriptors and assembling graphs...", flush=True)
    feature_graphs = []
    for record, unimol in zip(feature_records, unimol_by_record):
        properties = normalize_descriptors(
            descriptor_matrix(
                [residue.__dict__ for residue in record.residues]
            ),
            mean,
            std,
        )
        feature_graphs.append(
            make_graph(
                record,
                args.method,
                unimol,
                glycine,
                properties,
            )
        )

    query_graphs = feature_graphs[: len(valid_records)]
    reference_graph = feature_graphs[-1]
    query_graph_by_position = dict(zip(valid_positions, query_graphs))

    print(f"Predicting on {device}...", flush=True)
    model = load_model(checkpoint_path, device)
    output_rows = []
    for position, (row, record) in enumerate(zip(rows, records)):
        if record is None:
            output_rows.append(
                {
                    field: (
                        row["line_number"]
                        if field == "line_number"
                        else row["smiles"]
                        if field == "smiles"
                        else args.method
                        if field == "method"
                        else errors.get(int(row["line_number"]), "")
                        if field == "error"
                        else ""
                    )
                    for field in OUTPUT_FIELDS
                }
            )
            continue

        query_graph = query_graph_by_position[position]
        query_canonical = canonical_smiles(row["smiles"])
        if query_canonical == reference["canonical_smiles"]:
            delta_log = 0.0
        else:
            delta_log = predict_deltas(
                model,
                [query_graph],
                reference_graph,
                1,
                device,
            )[0]
        if reference["value_log10"] is None:
            absolute_log = ""
            absolute_linear = ""
        else:
            absolute_log_value = float(reference["value_log10"]) + delta_log
            absolute_log = f"{absolute_log_value:.3f}"
            absolute_linear = f"{10 ** (absolute_log_value + 6):.3f}"
        output_rows.append(
            {
                "line_number": row["line_number"],
                "smiles": row["smiles"],
                "method": args.method,
                "reference_source": reference_source,
                "reference_smiles": reference["smiles"],
                "delta_log10": f"{delta_log:.3f}",
                "absolute_log10_cm_s": absolute_log,
                "absolute_1e-6_cm_s": absolute_linear,
                "error": "",
            }
        )

    write_csv(output_rows, output_path)
    print(
        f"Complete: {len(valid_records)} queries, {len(errors)} failures; "
        f"output={output_path}",
        flush=True,
    )
    print(message, flush=True)


if __name__ == "__main__":
    main()
