#!/usr/bin/env python
"""Predict permeability deltas against the first SMILES in a text file."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    "sequence",
    "num_residues",
    "parent_line_number",
    "method",
    "prediction_delta",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "现场计算 SMILES 特征，并预测每一行相对第一行的 Δ 透膜性 "
            "(current - first)"
        )
    )
    parser.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "smiles.txt"),
        help="每个非空行一个环肽 SMILES；第一个非空行固定为 parent",
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
        "--device",
        default="auto",
        help="auto、cpu、cuda 或具体设备（如 cuda:0）",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unimol-batch-size", type=int, default=128)
    parser.add_argument(
        "--remove-hs",
        action="store_true",
        help="传给 Uni-Mol；默认保留氢，与当前训练特征一致",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境中 torch.cuda.is_available() 为 False")
    return device


def read_input(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    rows = [
        {"line_number": line_number, "smiles": text.strip()}
        for line_number, text in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        )
        if text.strip()
    ]
    if not rows:
        raise ValueError(f"输入文件不包含非空 SMILES: {path}")
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
    if records[0] is None:
        raise ValueError(
            "第一个非空 SMILES 必须能成功解析，因为它固定作为母核；"
            f"第 {rows[0]['line_number']} 行错误: {errors[rows[0]['line_number']]}"
        )
    return records, errors


def load_normalizer(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(payload["descriptor_names"])
    if names != tuple(DESCRIPTOR_NAMES):
        raise ValueError("descriptor_normalizer.json 的描述符顺序与代码不一致")
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    expected = (len(DESCRIPTOR_NAMES),)
    if tuple(mean.shape) != expected or tuple(std.shape) != expected:
        raise ValueError(
            f"标准化常量维度错误: mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("标准化常量包含非有限值")
    if not bool((std > 0).all()):
        raise ValueError("标准化 std 必须全部大于 0")
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
                raise ValueError(f"Uni-Mol 输出形状异常: {values.shape}")
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
            f"Uni-Mol 特征与残基数不一致: {tuple(unimol.shape)}, L={count}"
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
    for start in range(0, len(graphs), batch_size):
        children = graphs[start : start + batch_size]
        child_batch = move_graph(collate_graphs(children), device)
        child_embedding = model._pool_graph(child_batch, len(children))
        differences = child_embedding - parent_embedding.expand(len(children), -1)
        values = model.out_mlp(differences).squeeze(-1)
        predictions.extend(values.detach().cpu().tolist())
    return predictions


def build_output_rows(
    rows: list[dict[str, Any]],
    records: list[Any],
    errors: dict[int, str],
    prediction_by_line: dict[int, float],
    method: str,
) -> list[dict[str, Any]]:
    parent_line = int(rows[0]["line_number"])
    output = []
    for row, record in zip(rows, records):
        line_number = int(row["line_number"])
        error = errors.get(line_number, "")
        output.append(
            {
                "line_number": line_number,
                "smiles": row["smiles"],
                "sequence": "" if record is None else record.sequence,
                "num_residues": "" if record is None else len(record.residues),
                "parent_line_number": parent_line,
                "method": method,
                "prediction_delta": (
                    "" if record is None else prediction_by_line[line_number]
                ),
                "status": "preprocess_failed" if record is None else "ok",
                "error": error,
            }
        )
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.unimol_batch_size < 1:
        raise ValueError("所有 batch size 必须大于 0")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    normalizer_path = Path(args.normalizer).expanduser().resolve()
    device = resolve_device(args.device)

    rows = read_input(input_path)
    records, errors = parse_records(rows)
    valid_records = [record for record in records if record is not None]
    print(
        f"Parsed {len(valid_records)}/{len(rows)} SMILES; parent line="
        f"{rows[0]['line_number']}",
        flush=True,
    )

    mean, std = load_normalizer(normalizer_path)
    residue_counts = [len(record.residues) for record in valid_records]
    residue_smiles = [
        residue.smiles
        for record in valid_records
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
    graphs = []
    for record, unimol in zip(valid_records, unimol_by_record):
        properties = normalize_descriptors(
            descriptor_matrix(
                [residue.__dict__ for residue in record.residues]
            ),
            mean,
            std,
        )
        graphs.append(
            make_graph(
                record,
                args.method,
                unimol,
                glycine,
                properties,
            )
        )

    print(f"Predicting on {device}...", flush=True)
    model = load_model(checkpoint_path, device)
    predictions = predict_deltas(
        model,
        graphs,
        graphs[0],
        args.batch_size,
        device,
    )
    valid_lines = [
        int(row["line_number"])
        for row, record in zip(rows, records)
        if record is not None
    ]
    prediction_by_line = dict(zip(valid_lines, predictions))
    output_rows = build_output_rows(
        rows,
        records,
        errors,
        prediction_by_line,
        args.method,
    )
    write_csv(output_rows, output_path)
    print(
        f"Complete: {len(predictions)} predictions, {len(errors)} failures; "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
