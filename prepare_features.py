#!/usr/bin/env python
"""Generate all model features from the repository's raw tabular data."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from peptide_preprocessing import parse_cyclic_peptide
from process_monomer_descriptors import (
    DESCRIPTOR_NAMES,
    atomic_torch_save,
    descriptor_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
GLYCINE_MONOMER_ID = 52
GLYCINE_SMILES = "NCC(=O)O"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate peptide manifests, residue-level Uni-Mol embeddings, and "
            "monomer physicochemical descriptors."
        )
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PERMEABILITY_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
        help="Data directory containing CycPeptMPDB and data_split.",
    )
    parser.add_argument("--unimol-batch-size", type=int, default=128)
    parser.add_argument(
        "--remove-hs",
        action="store_true",
        help="Remove hydrogens in Uni-Mol. Leave disabled to match the released model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--unimol-only",
        action="store_true",
        help="Regenerate Uni-Mol tensors and leave the packaged descriptors in place.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    if "CAPA" in frame.columns and "Permeability_Value" not in frame.columns:
        frame["Permeability_Value"] = frame["CAPA"]
    if "Index" not in frame.columns:
        frame.insert(0, "Index", range(len(frame)))
    required = {"Index", "SMILES", "Permeability_Value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if path.suffix.lower() != ".xlsx" and "Method" not in frame.columns:
        raise ValueError(f"{path} is missing required column: Method")
    frame = frame.dropna(subset=list(required)).copy()
    frame["Index"] = frame["Index"].astype(int)
    if frame["Index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate Index values")
    return frame


def _create_manifest(table_path: Path, manifest_path: Path) -> dict:
    frame = _read_table(table_path)
    records = []
    errors = []
    for row in frame.itertuples(index=False):
        index = int(row.Index)
        smiles = str(row.SMILES)
        try:
            records.append(parse_cyclic_peptide(smiles, index).to_dict())
        except Exception as exc:
            errors.append({"index": index, "smiles": smiles, "error": str(exc)})
    manifest = {
        "format_version": 1,
        "mapping_method": "exact_achiral_structure",
        "source_file": str(table_path),
        "records": records,
        "errors": errors,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            f"Failed to preprocess {len(errors)} rows from {table_path}; "
            f"details were written to {manifest_path}"
        )
    return manifest


def _load_or_create_manifest(
    table_path: Path,
    manifest_path: Path,
    overwrite: bool,
) -> dict:
    _read_table(table_path)
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return _create_manifest(table_path, manifest_path)


def _encode(model, smiles: list[str], batch_size: int) -> torch.Tensor:
    batches = []
    for start in range(0, len(smiles), batch_size):
        values = np.asarray(
            model.get_repr(
                smiles[start : start + batch_size],
                return_atomic_reprs=True,
            )["cls_repr"],
            dtype=np.float32,
        )
        expected = (min(batch_size, len(smiles) - start), 512)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Unexpected Uni-Mol output shape {tuple(values.shape)}; "
                f"expected {expected}"
            )
        batches.append(torch.from_numpy(values))
    return torch.cat(batches, dim=0)


def _generate_dataset_features(
    model,
    manifest: dict,
    prefix: str,
    data_root: Path,
    batch_size: int,
    overwrite: bool,
    write_descriptors: bool,
) -> tuple[int, int]:
    unimol_dir = data_root / "unimol" / "residue_stack"
    descriptor_dir = data_root / "monomer_physicochemical"
    generated = skipped = 0
    for number, record in enumerate(manifest["records"], start=1):
        index = int(record["index"])
        filename = f"{prefix}{index}_our_process.pt" if not prefix else f"{prefix}{index}.pt"
        unimol_path = unimol_dir / filename
        descriptor_path = descriptor_dir / f"{index}_our_process.pt"
        needs_unimol = overwrite or not unimol_path.exists()
        needs_descriptors = write_descriptors and (
            overwrite or not descriptor_path.exists()
        )
        if not needs_unimol and not needs_descriptors:
            skipped += 1
            continue
        residues = record["residues"]
        if needs_unimol:
            embeddings = _encode(
                model,
                [str(residue["smiles"]) for residue in residues],
                batch_size,
            )
            atomic_torch_save(embeddings, unimol_path)
        if needs_descriptors:
            atomic_torch_save(descriptor_matrix(residues), descriptor_path)
        generated += 1
        print(
            f"[{number}/{len(manifest['records'])}] "
            f"{prefix or 'internal'} Index={index}",
            flush=True,
        )
    return generated, skipped


def main() -> None:
    args = parse_args()
    if args.unimol_batch_size < 1:
        raise ValueError("--unimol-batch-size must be positive")
    data_root = Path(args.data_root).expanduser().resolve()
    internal_table = data_root / "CycPeptMPDB" / "cycpept_mixed_converted.csv"
    internal_manifest_path = (
        data_root / "CycPeptMPDB" / "our_process_manifest.json"
    )
    merz_table = (
        data_root / "data_split" / "holdout_split" / "merz_data.xlsx"
    )
    merz_manifest_path = merz_table.with_name(
        "merz_data_our_process_manifest.json"
    )

    internal_manifest = _load_or_create_manifest(
        internal_table, internal_manifest_path, False
    )
    merz_manifest = _load_or_create_manifest(
        merz_table, merz_manifest_path, False
    )
    nielsen_table = (
        data_root / "data_split" / "holdout_split" / "Nielsen_data.xlsx"
    )
    nielsen_manifest_path = nielsen_table.with_name(
        "nielsen_data_our_process_manifest.json"
    )
    nielsen_manifest = _load_or_create_manifest(
        nielsen_table, nielsen_manifest_path, False
    )

    from unimol_tools import UniMolRepr

    model = UniMolRepr(data_type="molecule", remove_hs=args.remove_hs)
    try:
        glycine_path = (
            data_root
            / "unimol"
            / "monomer_cls"
            / f"{GLYCINE_MONOMER_ID}_our_process.pt"
        )
        if args.overwrite or not glycine_path.exists():
            atomic_torch_save(
                _encode(model, [GLYCINE_SMILES], args.unimol_batch_size)[0],
                glycine_path,
            )
        internal_counts = _generate_dataset_features(
            model,
            internal_manifest,
            "",
            data_root,
            args.unimol_batch_size,
            args.overwrite,
            write_descriptors=not args.unimol_only,
        )
        merz_counts = _generate_dataset_features(
            model,
            merz_manifest,
            "merz_data_",
            data_root,
            args.unimol_batch_size,
            args.overwrite,
            write_descriptors=False,
        )
        nielsen_counts = _generate_dataset_features(
            model,
            nielsen_manifest,
            "nielsen_data_",
            data_root,
            args.unimol_batch_size,
            args.overwrite,
            write_descriptors=False,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "descriptor_dimension": len(DESCRIPTOR_NAMES),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "remove_hs": bool(args.remove_hs),
        "internal_generated": internal_counts[0],
        "internal_skipped": internal_counts[1],
        "merz_data_generated": merz_counts[0],
        "merz_data_skipped": merz_counts[1],
        "nielsen_data_generated": nielsen_counts[0],
        "nielsen_data_skipped": nielsen_counts[1],
    }
    metadata_path = data_root / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Feature preparation complete: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
