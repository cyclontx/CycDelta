#!/usr/bin/env python
"""Calculate per-monomer physicochemical descriptors from the existing manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_SUFFIX = "_our_process.pt"
DESCRIPTOR_NAMES = (
    "MolWt",
    "ExactMolWt",
    "TPSA",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumRotatableBonds",
    "LabuteASA",
    "RadiusOfGyration",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "BertzCT",
    "BalabanJ",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Chi4v",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "HallKierAlpha",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate physicochemical descriptors for each peptide residue."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PERMEABILITY_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Defaults to <data-root>/CycPeptMPDB/our_process_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <data-root>/monomer_physicochemical.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files; by default completed files are skipped.",
    )
    return parser.parse_args()


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def calc_descriptors(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid monomer SMILES: {smiles}")

    mol3d = Chem.AddHs(mol)
    try:
        has_conformer = (
            AllChem.EmbedMolecule(
                mol3d,
                randomSeed=42,
                useRandomCoords=True,
            )
            == 0
        )
    except Exception:
        has_conformer = False
    if has_conformer:
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            pass

    values = {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "RingCount": Lipinski.RingCount(mol),
        "FractionCSP3": Lipinski.FractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "Chi0v": Descriptors.Chi0v(mol),
        "Chi1v": Descriptors.Chi1v(mol),
        "Chi2v": Descriptors.Chi2v(mol),
        "Chi3v": Descriptors.Chi3v(mol),
        "Chi4v": Descriptors.Chi4v(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
    }
    shape_functions = {
        "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
        "Asphericity": rdMolDescriptors.CalcAsphericity,
        "Eccentricity": rdMolDescriptors.CalcEccentricity,
        "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor,
        "NPR1": rdMolDescriptors.CalcNPR1,
        "NPR2": rdMolDescriptors.CalcNPR2,
        "PMI1": rdMolDescriptors.CalcPMI1,
        "PMI2": rdMolDescriptors.CalcPMI2,
        "PMI3": rdMolDescriptors.CalcPMI3,
    }
    for name, function in shape_functions.items():
        if not has_conformer:
            values[name] = float("nan")
            continue
        try:
            values[name] = function(mol3d)
        except Exception:
            values[name] = float("nan")

    return [float(values[name]) for name in DESCRIPTOR_NAMES]


def descriptor_matrix(residues: list[dict]) -> torch.Tensor:
    matrix = torch.tensor(
        [calc_descriptors(residue["smiles"]) for residue in residues],
        dtype=torch.float32,
    )
    expected = (len(residues), len(DESCRIPTOR_NAMES))
    if tuple(matrix.shape) != expected:
        raise ValueError(
            f"Descriptor shape {tuple(matrix.shape)} does not match expected {expected}"
        )
    return matrix


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else data_root / "CycPeptMPDB" / "our_process_manifest.json"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else data_root / "monomer_physicochemical"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    errors = []
    for number, record in enumerate(records, start=1):
        index = int(record["index"])
        output_path = output_dir / f"{index}{OUTPUT_SUFFIX}"
        if output_path.exists() and not args.overwrite:
            print(f"[{number}/{len(records)}] Index={index} skipped", flush=True)
            continue
        try:
            features = descriptor_matrix(record["residues"])
            atomic_torch_save(features, output_path)
            print(
                f"[{number}/{len(records)}] Index={index} shape={tuple(features.shape)}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"index": index, "error": str(exc)})
            print(
                f"[{number}/{len(records)}] Index={index} FAILED: {exc}",
                flush=True,
            )

    metadata = {
        "dimension": len(DESCRIPTOR_NAMES),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "source_manifest": str(manifest_path),
        "errors": errors,
    }
    (output_dir / "descriptor_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Complete: {len(records) - len(errors)} succeeded, {len(errors)} failed; "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
