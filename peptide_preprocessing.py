"""SMILES-to-residue preprocessing for cyclic peptide inference.

The trained model consumes a residue graph, not an atom graph.  This module
cuts only amide bonds that participate in a ring, hydrolyses the cut sites,
orders the resulting residues around the cycle, and maps a capped fragment
only when its structure exactly matches a natural amino acid (ignoring stereo).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem


AA_SMILES = {
    "A": "C[C@H](N)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O",
    "D": "N[C@@H](CC(=O)O)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O",
    "F": "N[C@@H](Cc1ccccc1)C(=O)O",
    "G": "NCC(=O)O",
    "H": "N[C@@H](Cc1c[nH]cn1)C(=O)O",
    "I": "CC[C@H](C)[C@@H](N)C(=O)O",
    "K": "NCCCC[C@@H](N)C(=O)O",
    "L": "CC(C)C[C@@H](N)C(=O)O",
    "M": "CSCC[C@@H](N)C(=O)O",
    "N": "NC(=O)C[C@@H](N)C(=O)O",
    "P": "O=C(O)[C@@H]1CCCN1",
    "Q": "NC(=O)CC[C@@H](N)C(=O)O",
    "R": "N=C(N)NCCC[C@@H](N)C(=O)O",
    "S": "N[C@@H](CO)C(=O)O",
    "T": "C[C@@H](O)[C@@H](N)C(=O)O",
    "V": "CC(C)[C@@H](N)C(=O)O",
    "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
}
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


REFERENCE_MOLS = {aa: _mol(smiles) for aa, smiles in AA_SMILES.items()}
REFERENCE_CANONICAL = {
    aa: Chem.MolToSmiles(mol, isomericSmiles=True)
    for aa, mol in REFERENCE_MOLS.items()
}


def _achiral_canonical_smiles(mol: Chem.Mol) -> str:
    """Return an exact molecular-graph key with all stereochemistry removed."""
    achiral = Chem.Mol(mol)
    Chem.RemoveStereochemistry(achiral)
    return Chem.MolToSmiles(achiral, canonical=True, isomericSmiles=False)


REFERENCE_ACHIRAL = {
    _achiral_canonical_smiles(mol): aa for aa, mol in REFERENCE_MOLS.items()
}


@dataclass
class ResidueRecord:
    position: int
    smiles: str
    amino_acid: str
    similarity: float
    is_natural: int
    is_D: int
    is_methylated: int


@dataclass
class PeptideRecord:
    index: int
    input_smiles: str
    canonical_smiles: str
    sequence: str
    helm_like: str
    cyclized_pair: list[int]
    residues: list[ResidueRecord]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["num_residues"] = len(self.residues)
        return result


def find_macrocyclic_amide_bonds(mol: Chem.Mol) -> list[tuple[int, int, int]]:
    """Return ``(bond_idx, carbonyl_C_idx, N_idx)`` for ring amide bonds."""
    found = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or not bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7:
            carbon, nitrogen = a, b
        elif a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6:
            carbon, nitrogen = b, a
        else:
            continue
        has_carbonyl_oxygen = any(
            neighbor.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(carbon.GetIdx(), neighbor.GetIdx()).GetBondType()
            == Chem.BondType.DOUBLE
            for neighbor in carbon.GetNeighbors()
        )
        if not has_carbonyl_oxygen:
            continue

        # SSSR is only a cycle basis and can omit the outer cycle in fused
        # systems.  Remove this bond and measure its actual alternate route.
        # A macrocyclic backbone amide has a long alternate C...N path, while
        # a side-chain lactam/urea ring has only a short local route.
        probe = Chem.RWMol(mol)
        probe.RemoveBond(carbon.GetIdx(), nitrogen.GetIdx())
        alternate_path = Chem.GetShortestPath(
            probe.GetMol(), carbon.GetIdx(), nitrogen.GetIdx()
        )
        if len(alternate_path) >= 8:
            found.append((bond.GetIdx(), carbon.GetIdx(), nitrogen.GetIdx()))
    if len(found) < 2:
        raise ValueError(
            f"Only {len(found)} macrocyclic amide bond(s) found; at least two are required"
        )
    return found


def _component_order(
    atom_to_component: dict[int, int],
    cut_bonds: Iterable[tuple[int, int, int]],
) -> tuple[list[int], bool]:
    components = set(atom_to_component.values())
    cut_bonds = list(cut_bonds)
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    directed_valid = True
    for _, carbon_idx, nitrogen_idx in cut_bonds:
        carbon_component = atom_to_component[carbon_idx]
        nitrogen_component = atom_to_component[nitrogen_idx]
        if (
            carbon_component == nitrogen_component
            or carbon_component in successor
            or nitrogen_component in predecessor
        ):
            directed_valid = False
            break
        successor[carbon_component] = nitrogen_component
        predecessor[nitrogen_component] = carbon_component

    if directed_valid and set(successor) == components and set(predecessor) == components:
        start = min(components)
        order = [start]
        while len(order) < len(components):
            next_component = successor[order[-1]]
            if next_component in order:
                directed_valid = False
                break
            order.append(next_component)
        if directed_valid and successor[order[-1]] == start:
            return order, False

    # Peptidomimetic linkers can reverse the local amide direction: one
    # fragment then has two N-facing cuts and another has none. The residue
    # connectivity can still be a valid simple cycle, so order it without
    # imposing N-to-C direction.
    adjacency = {component: set() for component in components}
    for _, carbon_idx, nitrogen_idx in cut_bonds:
        first = atom_to_component[carbon_idx]
        second = atom_to_component[nitrogen_idx]
        if first == second:
            raise ValueError("A selected amide cut did not separate two residues")
        adjacency[first].add(second)
        adjacency[second].add(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError(
            "Cut fragments do not form a simple residue cycle, even without N-to-C direction"
        )

    start = min(components)
    order = [start]
    previous = None
    current = start
    while len(order) < len(components):
        candidates = sorted(adjacency[current] - ({previous} if previous is not None else set()))
        next_component = next(
            (component for component in candidates if component not in order),
            None,
        )
        if next_component is None:
            raise ValueError("Undirected residue cycle closes before all fragments are visited")
        order.append(next_component)
        previous, current = current, next_component
    if start not in adjacency[order[-1]]:
        raise ValueError("Undirected residue order does not close into a cycle")
    return order, True


def hydrolyse_macrocycle(
    mol: Chem.Mol,
) -> tuple[list[Chem.Mol], list[tuple[int, tuple[int, ...]]]]:
    """Cut ring amides, add OH at each acyl end and H implicitly at each N end."""
    cuts = find_macrocyclic_amide_bonds(mol)
    editable = Chem.RWMol(mol)
    for _, carbon_idx, nitrogen_idx in cuts:
        editable.RemoveBond(carbon_idx, nitrogen_idx)
        oxygen_idx = editable.AddAtom(Chem.Atom(8))
        editable.AddBond(carbon_idx, oxygen_idx, Chem.BondType.SINGLE)

    hydrolysed = editable.GetMol()
    Chem.SanitizeMol(hydrolysed)
    fragments_atom_ids: list[tuple[int, ...]] = []
    fragments = list(
        Chem.GetMolFrags(
            hydrolysed,
            asMols=True,
            sanitizeFrags=True,
            fragsMolAtomMapping=fragments_atom_ids,
        )
    )
    if len(fragments) != len(cuts):
        raise ValueError(
            f"Expected {len(cuts)} residues after cutting, obtained {len(fragments)}"
        )

    atom_to_component = {}
    original_atom_count = mol.GetNumAtoms()
    for component, atom_ids in enumerate(fragments_atom_ids):
        for atom_idx in atom_ids:
            if atom_idx < original_atom_count:
                atom_to_component[atom_idx] = component
    order, used_undirected_fallback = _component_order(atom_to_component, cuts)

    # A standard residue has one backbone N. Direction-reversing linkers may
    # have zero or two; retain all of them so methylation can be detected
    # without rejecting an otherwise valid residue cycle.
    nitrogens_by_component: dict[int, list[int]] = {
        component: [] for component in order
    }
    for _, _, nitrogen_idx in cuts:
        component = atom_to_component[nitrogen_idx]
        nitrogens_by_component[component].append(nitrogen_idx)
    ordered_nitrogens = [
        (component, tuple(nitrogens_by_component[component])) for component in order
    ]
    ordered_fragments = [fragments[component] for component in order]

    # A cyclic sequence has no intrinsic first residue.  Canonicalize its
    # rotation so equivalent/randomized SMILES yield identical residue order.
    keys = [
        Chem.MolToSmiles(fragment, isomericSmiles=True)
        for fragment in ordered_fragments
    ]
    candidates = [
        (tuple(keys[offset:] + keys[:offset]), False, offset)
        for offset in range(len(keys))
    ]
    if used_undirected_fallback:
        reversed_keys = list(reversed(keys))
        candidates.extend(
            (tuple(reversed_keys[offset:] + reversed_keys[:offset]), True, offset)
            for offset in range(len(keys))
        )
    _, reverse_order, rotation = min(candidates)
    if reverse_order:
        ordered_fragments = list(reversed(ordered_fragments))
        ordered_nitrogens = list(reversed(ordered_nitrogens))
    ordered_fragments = (
        ordered_fragments[rotation:] + ordered_fragments[:rotation]
    )
    ordered_nitrogens = (
        ordered_nitrogens[rotation:] + ordered_nitrogens[:rotation]
    )
    return ordered_fragments, ordered_nitrogens


def map_fragment(fragment: Chem.Mol) -> tuple[str, float, str]:
    """Map only an exact natural-AA structure, ignoring all stereochemistry."""
    amino_acid = REFERENCE_ACHIRAL.get(_achiral_canonical_smiles(fragment), "X")
    match_score = float(amino_acid != "X")
    return amino_acid, match_score, amino_acid


def _backbone_n_is_methylated(mol: Chem.Mol, nitrogen_idx: int) -> int:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    terminal_carbon_neighbors = 0
    for neighbor in nitrogen.GetNeighbors():
        if neighbor.GetAtomicNum() != 6:
            continue
        heavy_neighbors = [
            atom for atom in neighbor.GetNeighbors() if atom.GetAtomicNum() > 1
        ]
        if len(heavy_neighbors) == 1:
            terminal_carbon_neighbors += 1
    return int(terminal_carbon_neighbors > 0)


def _alpha_cip(fragment: Chem.Mol) -> str | None:
    Chem.AssignStereochemistry(fragment, cleanIt=True, force=True)
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        neighbors = list(atom.GetNeighbors())
        has_n = any(n.GetAtomicNum() == 7 for n in neighbors)
        has_carboxyl = any(
            n.GetAtomicNum() == 6
            and any(
                o.GetAtomicNum() == 8
                and fragment.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx()).GetBondType()
                == Chem.BondType.DOUBLE
                for o in n.GetNeighbors()
            )
            for n in neighbors
        )
        if has_n and has_carboxyl and atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
    return None


def _is_d(fragment: Chem.Mol, reference_amino_acid: str) -> tuple[int, int]:
    if reference_amino_acid in ("G", "X"):
        return 0, 0
    observed = _alpha_cip(fragment)
    reference = _alpha_cip(REFERENCE_MOLS[reference_amino_acid])
    assigned = int(bool(observed and reference))
    return int(bool(assigned and observed != reference)), assigned


def residue_feature_flags(
    amino_acid: str,
    is_d: int,
    stereo_assigned: int,
    has_n_methyl: int,
) -> tuple[int, int, int]:
    """Return model flags as ``(is_natural_L, is_D, is_methylated)``."""
    backbone_is_natural = amino_acid != "X"
    is_methylated = int(backbone_is_natural and has_n_methyl)
    is_d = int(backbone_is_natural and is_d)
    is_natural = int(
        backbone_is_natural
        and not is_methylated
        and not is_d
        and (stereo_assigned or amino_acid == "G")
    )
    return is_natural, is_d, is_methylated


def parse_cyclic_peptide(
    smiles: str,
    index: int,
) -> PeptideRecord:
    mol = _mol(smiles)
    fragments, ordered_nitrogens = hydrolyse_macrocycle(mol)
    residues = []
    for position, (fragment, (_, original_nitrogen_indices)) in enumerate(
        zip(fragments, ordered_nitrogens), start=1
    ):
        fragment_smiles = Chem.MolToSmiles(fragment, isomericSmiles=True)
        amino_acid, similarity, best_amino_acid = map_fragment(fragment)
        is_d, stereo_assigned = _is_d(fragment, best_amino_acid)
        # Feature semantics used by the trained model:
        #   is_natural: unmodified natural amino acid in the L configuration
        #   is_D:       natural-amino-acid identity in the D configuration
        # Exact matching makes every modified residue X, so its flags are zero.
        # Gly is achiral and is therefore treated as L without a CIP assignment.
        is_natural, is_d, is_methylated = residue_feature_flags(
            amino_acid,
            is_d,
            stereo_assigned,
            int(
                any(
                    _backbone_n_is_methylated(mol, nitrogen_idx)
                    for nitrogen_idx in original_nitrogen_indices
                )
            ),
        )
        residues.append(
            ResidueRecord(
                position=position,
                smiles=fragment_smiles,
                amino_acid=amino_acid,
                similarity=similarity,
                is_natural=is_natural,
                is_D=is_d,
                is_methylated=is_methylated,
            )
        )

    sequence = "".join(residue.amino_acid for residue in residues)
    helm_tokens = []
    for residue in residues:
        flags = []
        if residue.is_D:
            flags.append("D")
        if residue.is_methylated:
            flags.append("NMe")
        label = residue.amino_acid
        helm_tokens.append("-".join(flags + [label]) if flags else label)
    helm_like = (
        f"PEPTIDE1{{{'.'.join(helm_tokens)}}}$"
        f"PEPTIDE1,PEPTIDE1,1:R1-{len(residues)}:R2$$$"
    )
    return PeptideRecord(
        index=index,
        input_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
        sequence=sequence,
        helm_like=helm_like,
        cyclized_pair=[0, len(residues) - 1],
        residues=residues,
    )


def preprocess_smiles_file(
    input_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[PeptideRecord], list[dict]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_lines = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[PeptideRecord] = []
    errors: list[dict] = []
    for index, smiles in enumerate(smiles_lines, start=1):
        try:
            records.append(parse_cyclic_peptide(smiles, index))
        except Exception as exc:
            errors.append({"index": index, "smiles": smiles, "error": str(exc)})

    manifest = {
        "format_version": 1,
        "mapping_method": "exact_achiral_structure",
        "source_file": str(input_path),
        "records": [record.to_dict() for record in records],
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "residue_mapping.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "position",
                "fragment_smiles",
                "amino_acid",
                "similarity",
                "is_natural",
                "is_D",
                "is_methylated",
            ],
        )
        writer.writeheader()
        for record in records:
            for residue in record.residues:
                writer.writerow(
                    {
                        "index": record.index,
                        "position": residue.position,
                        "fragment_smiles": residue.smiles,
                        "amino_acid": residue.amino_acid,
                        "similarity": f"{residue.similarity:.6f}",
                        "is_natural": residue.is_natural,
                        "is_D": residue.is_D,
                        "is_methylated": residue.is_methylated,
                    }
                )
    if errors:
        with (output_dir / "preprocess_errors.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "smiles", "error"])
            writer.writeheader()
            writer.writerows(errors)
    return records, errors


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported manifest format: {manifest.get('format_version')}")
    return manifest
