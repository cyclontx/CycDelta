#!/usr/bin/env python
"""Build train-only or all-split monomer candidate libraries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from peptide_preprocessing import AA_SMILES  # noqa: E402
from process_monomer_descriptors import (  # noqa: E402
    DESCRIPTOR_NAMES,
    calc_descriptors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a monomer candidate library from the full database."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--candidate-scope",
        choices=("train", "all"),
        default="all",
        help=(
            "all=use the complete database manifest (default); "
            "train=restrict candidates to OD_train"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Defaults to candidate_library.pt for all scope or "
            "candidate_library_train.pt for train scope."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--remove-hs", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> set[int]:
    return {
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid monomer SMILES: {smiles}")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def detect_n_methylated(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    for alpha_carbon in molecule.GetAtoms():
        if alpha_carbon.GetAtomicNum() != 6:
            continue
        neighbors = list(alpha_carbon.GetNeighbors())
        nitrogens = [
            atom for atom in neighbors if atom.GetAtomicNum() == 7
        ]
        has_carboxyl = any(
            atom.GetAtomicNum() == 6
            and any(
                oxygen.GetAtomicNum() == 8
                and molecule.GetBondBetweenAtoms(
                    atom.GetIdx(), oxygen.GetIdx()
                ).GetBondType()
                == Chem.BondType.DOUBLE
                for oxygen in atom.GetNeighbors()
            )
            for atom in neighbors
        )
        if not nitrogens or not has_carboxyl:
            continue
        for nitrogen in nitrogens:
            for neighbor in nitrogen.GetNeighbors():
                if (
                    neighbor.GetIdx() == alpha_carbon.GetIdx()
                    or neighbor.GetAtomicNum() != 6
                ):
                    continue
                heavy_neighbors = [
                    atom
                    for atom in neighbor.GetNeighbors()
                    if atom.GetAtomicNum() > 1
                ]
                if len(heavy_neighbors) == 1:
                    return 1
    return 0


def modification_class(candidate: dict) -> str:
    if candidate["is_natural"]:
        return "natural_L"
    if candidate["is_D"]:
        return "D_amino_acid"
    if candidate["detected_n_methylated"]:
        return "N_methylated"
    return "non_natural"


def collect_candidates(
    manifest: dict,
    candidate_scope: str,
    train_ids: set[int],
) -> list[dict]:
    candidates_by_smiles = {}
    frequencies: Counter[str] = Counter()
    for record in manifest["records"]:
        if (
            candidate_scope == "train"
            and int(record["index"]) not in train_ids
        ):
            continue
        for residue in record["residues"]:
            smiles = canonical_smiles(residue["smiles"])
            frequencies[smiles] += 1
            candidate = {
                "smiles": smiles,
                "amino_acid": str(residue["amino_acid"]),
                "is_natural": int(residue["is_natural"]),
                "is_D": int(residue["is_D"]),
                # Preserve the exact flag semantics consumed during training.
                "is_methylated": int(residue["is_methylated"]),
                "detected_n_methylated": detect_n_methylated(smiles),
                "source": (
                    "training_manifest"
                    if candidate_scope == "train"
                    else "all_split_manifest"
                ),
            }
            previous = candidates_by_smiles.get(smiles)
            if previous is None:
                candidates_by_smiles[smiles] = candidate
                continue
            fields = (
                "amino_acid",
                "is_natural",
                "is_D",
                "is_methylated",
                "detected_n_methylated",
            )
            if any(previous[field] != candidate[field] for field in fields):
                raise ValueError(
                    f"Inconsistent annotations for monomer {smiles}"
                )

    # Always expose the complete standard L-amino-acid set.
    for amino_acid, raw_smiles in AA_SMILES.items():
        smiles = canonical_smiles(raw_smiles)
        candidates_by_smiles.setdefault(
            smiles,
            {
                "smiles": smiles,
                "amino_acid": amino_acid,
                "is_natural": 1,
                "is_D": 0,
                "is_methylated": 0,
                "detected_n_methylated": 0,
                "source": "standard_amino_acids",
            },
        )

    output = []
    for smiles, candidate in sorted(candidates_by_smiles.items()):
        candidate = dict(candidate)
        candidate["frequency"] = int(frequencies[smiles])
        candidate["modification_class"] = modification_class(candidate)
        output.append(candidate)
    return output


def unimol_representations(
    model,
    smiles: list[str],
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        values = np.asarray(
            model.get_repr(batch, return_atomic_reprs=True)["cls_repr"],
            dtype=np.float32,
        )
        if tuple(values.shape) != (len(batch), 512):
            raise ValueError(f"Unexpected Uni-Mol shape: {values.shape}")
        chunks.append(torch.from_numpy(values))
        print(
            f"Uni-Mol: {min(start + len(batch), len(smiles))}/"
            f"{len(smiles)}",
            flush=True,
        )
    return torch.cat(chunks)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = (
        data_root / "CycPeptMPDB" / "our_process_manifest.json"
    )
    train_path = (
        data_root / "data_split" / "holdout_split" / "OD_train.txt"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = read_ids(train_path)
    candidates = collect_candidates(
        manifest,
        args.candidate_scope,
        train_ids,
    )
    smiles = [candidate["smiles"] for candidate in candidates]

    from unimol_tools import UniMolRepr

    model = UniMolRepr(
        data_type="molecule",
        remove_hs=args.remove_hs,
    )
    unimol = unimol_representations(model, smiles, args.batch_size)
    descriptors = torch.tensor(
        [calc_descriptors(value) for value in smiles],
        dtype=torch.float32,
    )

    default_name = (
        "candidate_library_train.pt"
        if args.candidate_scope == "train"
        else "candidate_library.pt"
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else SCRIPT_DIR / default_name
    )
    torch.save(
        {
            "format_version": 2,
            "candidate_scope": args.candidate_scope,
            "manifest": str(manifest_path),
            "train_ids": str(train_path),
            "remove_hs": bool(args.remove_hs),
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "candidates": candidates,
            "unimol": unimol,
            "descriptors": descriptors,
        },
        output,
    )
    class_counts = Counter(
        candidate["modification_class"] for candidate in candidates
    )
    print(
        f"Saved {len(candidates)} {args.candidate_scope!r}-scope "
        f"candidates to {output}: {dict(class_counts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
