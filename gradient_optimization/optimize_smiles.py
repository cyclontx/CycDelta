#!/usr/bin/env python
"""Generate gradient-guided monomer substitutions for one cyclic peptide."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

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
from chemistry import assemble_validated_cycle, find_direction  # noqa: E402


METHODS = ("PAMPA", "CACO2", "MDCK", "RRCK")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_VOCAB)}
UNK_INDEX = len(AA_VOCAB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入一个环肽 SMILES，输出梯度筛选并完整复算的单体替换建议"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles")
    input_group.add_argument(
        "--input-result",
        help="从已有 optimization_result.json 复用输入 SMILES",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-library", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method", choices=METHODS, default="PAMPA")
    parser.add_argument(
        "--direction",
        choices=("increase", "decrease"),
        default="increase",
    )
    parser.add_argument(
        "--candidate-type",
        choices=("all", "natural", "non-natural"),
        default="all",
    )
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument(
        "--screen-top-k",
        type=int,
        default=200,
        help="梯度初筛后进行完整模型复算的数量",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="最终保存的最佳建议数量，默认 50",
    )
    parser.add_argument(
        "--screen-per-position",
        type=int,
        default=30,
        help="梯度初筛时每个位置至少保留的候选数",
    )
    parser.add_argument(
        "--min-per-position",
        type=int,
        default=5,
        help="最终结果中每个残基位置至少保留的建议数，默认 5",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "optimization_result.json"),
    )
    parser.add_argument("--d-emb", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-gnn-layer", type=int, default=2)
    return parser.parse_args()


def read_ids(path: Path) -> list[int]:
    return [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def monomer_normalizer(data_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    train_ids = read_ids(
        data_root / "data_split" / "holdout_split" / "OD_train.txt"
    )
    feature_dir = data_root / "monomer_physicochemical"
    matrices = [
        torch.load(
            feature_dir / f"{index}_our_process.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for index in train_ids
    ]
    features = torch.cat(matrices)
    features = features.masked_fill(~torch.isfinite(features), float("nan"))
    mean = torch.nanmean(features, dim=0)
    finite = torch.isfinite(features)
    centered = torch.where(finite, features - mean, torch.zeros_like(features))
    std = torch.sqrt(
        centered.square().sum(dim=0) / finite.sum(dim=0).clamp_min(1)
    )
    std = torch.where(
        torch.isfinite(std) & (std >= 1e-6),
        std,
        torch.ones_like(std),
    )
    return mean, std


def load_model(args: argparse.Namespace, device: torch.device) -> CycGNN:
    model = CycGNN(
        d_emb=args.d_emb,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_gnn_layer=args.num_gnn_layer,
    )
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state or state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_edges(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    edges = []
    attributes = []
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
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attributes, dtype=torch.float32),
    )


def node_features(record, method: str) -> torch.Tensor:
    features = torch.zeros((len(record.residues), 28), dtype=torch.float32)
    for position, residue in enumerate(record.residues):
        features[position, :3] = torch.tensor(
            [residue.is_natural, residue.is_D, residue.is_methylated]
        )
        aa_index = AA_TO_INDEX.get(residue.amino_acid, UNK_INDEX)
        features[position, 3 + aa_index] = 1
        features[position, 24 + METHODS.index(method)] = 1
    return features


class RuntimeEncoders:
    """Compute Uni-Mol residue embeddings and release the encoder afterwards."""

    def __init__(self, remove_hs: bool):
        self.remove_hs = remove_hs

    def unimol_batch(self, smiles: list[str]) -> torch.Tensor:
        from unimol_tools import UniMolRepr

        model = UniMolRepr(
            data_type="molecule",
            remove_hs=self.remove_hs,
        )
        values = np.asarray(
            model.get_repr(smiles, return_atomic_reprs=True)["cls_repr"],
            dtype=np.float32,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tuple(values.shape) != (len(smiles), 512):
            raise ValueError(f"Unexpected Uni-Mol shape: {values.shape}")
        return torch.from_numpy(values)


def normalize_descriptors(
    values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    values = torch.where(torch.isfinite(values), values, mean)
    return (values - mean) / std


def make_graph(
    record,
    method: str,
    unimol: torch.Tensor,
    glycine: torch.Tensor,
    monomer_property: torch.Tensor,
) -> dict:
    edge_index, edge_attr = build_edges(len(record.residues))
    return {
        "index": torch.tensor([0]),
        "node_feature": node_features(record, method),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "delta_unimol_feature": unimol - glycine.unsqueeze(0),
        "monomer_property_feature": monomer_property,
        "node_batch": torch.zeros(len(record.residues), dtype=torch.long),
    }


def graph_to(
    graph: dict,
    device: torch.device,
    requires_grad: bool = False,
) -> dict:
    differentiable = {
        "node_feature",
        "delta_unimol_feature",
        "monomer_property_feature",
    }
    result = {}
    for key, value in graph.items():
        value = value.detach().clone().to(device)
        if requires_grad and key in differentiable:
            value.requires_grad_(True)
        result[key] = value
    return result


def pair_batch(child: dict, parent: dict, device: torch.device) -> dict:
    return {
        "child": child,
        "parent": parent,
        "index": torch.tensor([0], device=device),
        "parent_index": torch.tensor([0], device=device),
        "y": torch.zeros(1, device=device),
    }


def replacement_node(candidate: dict, current: torch.Tensor) -> torch.Tensor:
    value = current.detach().clone()
    value[:24] = 0
    value[:3] = torch.tensor(
        [
            candidate["is_natural"],
            candidate["is_D"],
            candidate["is_methylated"],
        ],
        dtype=torch.float32,
    )
    value[
        3 + AA_TO_INDEX.get(candidate["amino_acid"], UNK_INDEX)
    ] = 1
    return value


def candidate_mask(
    candidates: list[dict],
    candidate_type: str,
    min_frequency: int,
) -> torch.Tensor:
    mask = []
    for candidate in candidates:
        natural = candidate["modification_class"] == "natural_L"
        type_ok = (
            candidate_type == "all"
            or (candidate_type == "natural" and natural)
            or (candidate_type == "non-natural" and not natural)
        )
        frequency_ok = natural or candidate["frequency"] >= min_frequency
        mask.append(type_ok and frequency_ok)
    return torch.tensor(mask, dtype=torch.bool)


def screen(
    graph: dict,
    record,
    gradients: dict,
    library: dict,
    glycine: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    valid_mask: torch.Tensor,
    sign: float,
) -> list[dict]:
    candidate_unimol = library["unimol"].float()
    candidate_property = normalize_descriptors(
        library["descriptors"].float(),
        mean,
        std,
    )
    rows = []
    for position, residue in enumerate(record.residues):
        current_node = graph["node_feature"][position]
        nodes = torch.stack(
            [
                replacement_node(candidate, current_node)
                for candidate in library["candidates"]
            ]
        )
        score = sign * (
            (nodes - current_node)
            .mul(gradients["node_feature"][position].cpu())
            .sum(dim=1)
            + (
                candidate_unimol
                - graph["delta_unimol_feature"][position]
                - glycine
            )
            .mul(gradients["delta_unimol_feature"][position].cpu())
            .sum(dim=1)
            + (
                candidate_property
                - graph["monomer_property_feature"][position]
            )
            .mul(gradients["monomer_property_feature"][position].cpu())
            .sum(dim=1)
        )
        score = score.masked_fill(
            ~valid_mask[position],
            float("-inf"),
        )
        for candidate_index in torch.nonzero(
            torch.isfinite(score)
        ).flatten().tolist():
            candidate = library["candidates"][candidate_index]
            if candidate["smiles"] == residue.smiles:
                continue
            rows.append(
                {
                    "position": position,
                    "candidate_index": candidate_index,
                    "gradient_score": float(score[candidate_index]),
                }
            )
    return sorted(
        rows,
        key=lambda row: row["gradient_score"],
        reverse=True,
    )


def chemically_valid_mask(
    record,
    library: dict,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    monomers = [residue.smiles for residue in record.residues]
    direction = find_direction(record.input_smiles, monomers)
    valid = torch.zeros(
        (len(monomers), len(library["candidates"])),
        dtype=torch.bool,
    )
    for position in range(len(monomers)):
        for candidate_index in torch.nonzero(base_mask).flatten().tolist():
            candidate = library["candidates"][candidate_index]
            if candidate["smiles"] == monomers[position]:
                continue
            modified = list(monomers)
            modified[position] = candidate["smiles"]
            try:
                assemble_validated_cycle(modified, direction)
            except Exception:
                continue
            valid[position, candidate_index] = True
    counts = valid.sum(dim=1).tolist()
    if any(count == 0 for count in counts):
        raise ValueError(
            f"化学合理性过滤后部分位置没有候选: {counts}"
        )
    print(f"Chemically valid candidates per position: {counts}", flush=True)
    return valid


def reserve_per_position(
    rows: list[dict],
    total: int,
    minimum: int,
    num_positions: int,
) -> list[dict]:
    selected = []
    selected_keys = set()
    for position in range(num_positions):
        position_rows = [
            row for row in rows if row["position"] == position
        ][:minimum]
        for row in position_rows:
            key = (row["position"], row["candidate_index"])
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)
    for row in rows:
        if len(selected) >= total:
            break
        key = (row["position"], row["candidate_index"])
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    return sorted(
        selected,
        key=lambda row: row["gradient_score"],
        reverse=True,
    )


def select_diverse_exact(
    rows: list[dict],
    total: int,
    minimum: int,
    num_positions: int,
    sign: float,
) -> list[dict]:
    selected = []
    selected_keys = set()
    for position in range(1, num_positions + 1):
        position_rows = [
            row for row in rows if row["position"] == position
        ][:minimum]
        for row in position_rows:
            key = (row["position"], row["to_fragment_smiles"])
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)
    for row in rows:
        if len(selected) >= total:
            break
        key = (row["position"], row["to_fragment_smiles"])
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    selected.sort(
        key=lambda row: sign * row["predicted_permeability_delta"],
        reverse=True,
    )
    return selected[:total]


def gradient_summaries(
    record,
    gradients: dict,
    descriptor_std: torch.Tensor,
) -> list[dict]:
    raw_gradient = (
        gradients["monomer_property_feature"].cpu()
        / descriptor_std.unsqueeze(0)
    )
    rows = []
    for position, residue in enumerate(record.residues):
        values = raw_gradient[position]
        strongest = torch.argsort(values.abs(), descending=True)[:5]
        rows.append(
            {
                "position": position + 1,
                "current_amino_acid": residue.amino_acid,
                "current_fragment_smiles": residue.smiles,
                "descriptor_directions": [
                    {
                        "name": DESCRIPTOR_NAMES[int(index)],
                        "direction_for_higher_prediction": (
                            "increase" if values[index] > 0 else "decrease"
                        ),
                        "gradient": float(values[index]),
                    }
                    for index in strongest
                ],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.screen_top_k < args.top_k:
        raise ValueError("需要满足 screen-top-k >= top-k >= 1")
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    library = torch.load(
        Path(args.candidate_library).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if library["descriptor_names"] != list(DESCRIPTOR_NAMES):
        raise ValueError("候选库描述符顺序与当前代码不一致")
    candidate_scope = library.get("candidate_scope", "train")
    print(
        f"Candidate library: scope={candidate_scope}, "
        f"size={len(library['candidates'])}",
        flush=True,
    )

    smiles = args.smiles
    if args.input_result:
        previous_result = json.loads(
            Path(args.input_result)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
        smiles = previous_result["input"]["smiles"]
    record = parse_cyclic_peptide(smiles, index=0)
    num_positions = len(record.residues)
    if args.min_per_position * num_positions > args.top_k:
        raise ValueError(
            "min-per-position × 残基数不能大于 top-k"
        )
    if args.screen_per_position * num_positions > args.screen_top_k:
        raise ValueError(
            "screen-per-position × 残基数不能大于 screen-top-k"
        )
    base_candidate_mask = candidate_mask(
        library["candidates"],
        args.candidate_type,
        args.min_frequency,
    )
    valid_candidate_mask = chemically_valid_mask(
        record,
        library,
        base_candidate_mask,
    )
    mean, std = monomer_normalizer(data_root)
    model = load_model(args, device)
    encoders = RuntimeEncoders(bool(library["remove_hs"]))
    residue_smiles = [residue.smiles for residue in record.residues]
    unimol = encoders.unimol_batch(residue_smiles + ["NCC(=O)O"])
    glycine = unimol[-1]
    unimol = unimol[:-1]
    properties = normalize_descriptors(
        descriptor_matrix(
            [residue.__dict__ for residue in record.residues]
        ),
        mean,
        std,
    )
    graph = make_graph(
        record,
        args.method,
        unimol,
        glycine,
        properties,
    )

    parent = graph_to(graph, device)
    child = graph_to(graph, device, requires_grad=True)
    baseline_tensor = model(pair_batch(child, parent, device))[0]
    baseline_tensor.backward()
    feature_names = (
        "node_feature",
        "delta_unimol_feature",
        "monomer_property_feature",
    )
    gradients = {
        name: child[name].grad.detach()
        for name in feature_names
    }
    baseline = float(baseline_tensor.detach().cpu())
    sign = 1.0 if args.direction == "increase" else -1.0
    screened = reserve_per_position(
        screen(
            graph,
            record,
            gradients,
            library,
            glycine,
            mean,
            std,
            valid_candidate_mask,
            sign,
        ),
        args.screen_top_k,
        args.screen_per_position,
        num_positions,
    )
    if len(screened) < args.top_k:
        raise ValueError(
            f"筛选后只有 {len(screened)} 条建议，少于 top-k={args.top_k}"
        )

    exact = []
    with torch.inference_mode():
        for row in screened:
            position = row["position"]
            candidate_index = row["candidate_index"]
            candidate = library["candidates"][candidate_index]
            variant = graph_to(graph, torch.device("cpu"))
            variant["node_feature"][position] = replacement_node(
                candidate,
                variant["node_feature"][position],
            )
            variant["delta_unimol_feature"][position] = (
                library["unimol"][candidate_index].float() - glycine
            )
            variant["monomer_property_feature"][position] = (
                normalize_descriptors(
                    library["descriptors"][candidate_index].float(),
                    mean,
                    std,
                )
            )
            prediction = float(
                model(
                    pair_batch(
                        graph_to(variant, device),
                        parent,
                        device,
                    )
                )[0].cpu()
            )
            exact.append(
                {
                    "position": position + 1,
                    "from_amino_acid": record.residues[position].amino_acid,
                    "from_fragment_smiles": record.residues[position].smiles,
                    "to_amino_acid": candidate["amino_acid"],
                    "to_fragment_smiles": candidate["smiles"],
                    "modification_class": candidate["modification_class"],
                    "training_frequency": candidate["frequency"],
                    "gradient_score": row["gradient_score"],
                    "predicted_permeability_delta": prediction,
                    "improvement_over_identity": prediction - baseline,
                }
            )
    exact.sort(
        key=lambda row: sign * row["predicted_permeability_delta"],
        reverse=True,
    )
    selected = select_diverse_exact(
        exact,
        args.top_k,
        args.min_per_position,
        num_positions,
        sign,
    )
    output = {
        "input": {
            "smiles": smiles,
            "canonical_smiles": record.canonical_smiles,
            "sequence": record.sequence,
            "num_residues": len(record.residues),
            "method": args.method,
        },
        "settings": {
            "direction": args.direction,
            "candidate_type": args.candidate_type,
            "min_frequency": args.min_frequency,
            "screen_top_k": args.screen_top_k,
            "top_k": args.top_k,
            "screen_per_position": args.screen_per_position,
            "min_per_position": args.min_per_position,
            "candidate_scope": candidate_scope,
            "candidate_library_size": len(library["candidates"]),
        },
        "identity_pair_prediction": baseline,
        "residue_gradients": gradient_summaries(record, gradients, std),
        "suggestions": selected,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {args.top_k} suggestions to: {output_path}")


if __name__ == "__main__":
    main()
