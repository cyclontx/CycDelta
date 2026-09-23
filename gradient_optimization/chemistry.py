"""Chemical assembly and validation shared by optimization and plotting."""

from __future__ import annotations

from collections import Counter

from rdkit import Chem

from peptide_preprocessing import parse_cyclic_peptide


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def is_carbonyl(molecule: Chem.Mol, atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        neighbor.GetAtomicNum() == 8
        and molecule.GetBondBetweenAtoms(
            atom.GetIdx(), neighbor.GetIdx()
        ).GetBondType()
        == Chem.BondType.DOUBLE
        for neighbor in atom.GetNeighbors()
    )


def select_handles(molecule: Chem.Mol) -> tuple[int, int, int]:
    """Select an unambiguous peptide-forming carboxyl C/O and amino N."""
    carboxyls = []
    for carbon in molecule.GetAtoms():
        if not is_carbonyl(molecule, carbon):
            continue
        for oxygen in carbon.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(
                carbon.GetIdx(), oxygen.GetIdx()
            )
            if (
                oxygen.GetAtomicNum() == 8
                and oxygen.GetDegree() == 1
                and oxygen.GetFormalCharge() in (0, -1)
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                carboxyls.append((carbon.GetIdx(), oxygen.GetIdx()))
    nitrogens = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and atom.GetDegree() <= 2
        and not any(
            is_carbonyl(molecule, neighbor)
            for neighbor in atom.GetNeighbors()
        )
    ]
    if not carboxyls or not nitrogens:
        raise ValueError("No available amino/carboxyl condensation handles")

    choices = []
    for carbon, oxygen in carboxyls:
        for nitrogen in nitrogens:
            distance = len(
                Chem.GetShortestPath(molecule, carbon, nitrogen)
            )
            choices.append((distance, carbon, oxygen, nitrogen))
    choices.sort()
    minimum_distance = choices[0][0]
    best = [choice for choice in choices if choice[0] == minimum_distance]
    unique_pairs = {(choice[1], choice[3]) for choice in best}
    if len(unique_pairs) != 1:
        raise ValueError("Ambiguous amino/carboxyl condensation handles")
    _, carbon, oxygen, nitrogen = best[0]
    return carbon, oxygen, nitrogen


def prepare_monomer(smiles: str) -> tuple[Chem.Mol, int, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid monomer SMILES: {smiles}")
    carbon, oxygen, nitrogen = select_handles(molecule)
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(oxygen)
    carbon -= int(carbon > oxygen)
    nitrogen -= int(nitrogen > oxygen)
    return editable.GetMol(), carbon, nitrogen


def assemble_cycle(
    monomer_smiles: list[str],
    direction: int,
) -> tuple[Chem.Mol, list[list[int]], list[int]]:
    prepared = [prepare_monomer(smiles) for smiles in monomer_smiles]
    combined = None
    carbons = []
    nitrogens = []
    residue_atoms = []
    offset = 0
    for molecule, carbon, nitrogen in prepared:
        combined = (
            Chem.Mol(molecule)
            if combined is None
            else Chem.CombineMols(combined, molecule)
        )
        residue_atoms.append(
            list(range(offset, offset + molecule.GetNumAtoms()))
        )
        carbons.append(offset + carbon)
        nitrogens.append(offset + nitrogen)
        offset += molecule.GetNumAtoms()

    editable = Chem.RWMol(combined)
    added_bonds = []
    for position, carbon in enumerate(carbons):
        target = (position + direction) % len(prepared)
        editable.AddBond(
            carbon,
            nitrogens[target],
            Chem.BondType.SINGLE,
        )
        added_bonds.append(editable.GetNumBonds() - 1)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Assembled structure is disconnected")
    Chem.GetSymmSSSR(molecule)
    if not all(molecule.GetBondWithIdx(index).IsInRing() for index in added_bonds):
        raise ValueError("One or more peptide-forming bonds are not in the macrocycle")
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule, residue_atoms, added_bonds


def molecule_key(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def fragment_multiset(smiles: list[str]) -> Counter:
    return Counter(canonical_smiles(value) for value in smiles)


def validate_rehydrolysis(
    molecule: Chem.Mol,
    expected_monomers: list[str],
) -> None:
    record = parse_cyclic_peptide(molecule_key(molecule), index=0)
    observed = fragment_multiset(
        [residue.smiles for residue in record.residues]
    )
    expected = fragment_multiset(expected_monomers)
    if observed != expected:
        raise ValueError(
            "Re-hydrolysis does not recover the intended monomer set"
        )


def assemble_validated_cycle(
    monomer_smiles: list[str],
    direction: int,
) -> tuple[Chem.Mol, list[list[int]]]:
    molecule, residue_atoms, _ = assemble_cycle(
        monomer_smiles,
        direction,
    )
    validate_rehydrolysis(molecule, monomer_smiles)
    return molecule, residue_atoms


def find_direction(
    input_smiles: str,
    monomer_smiles: list[str],
) -> int:
    expected = canonical_smiles(input_smiles)
    errors = []
    for direction in (1, -1):
        try:
            molecule, _, _ = assemble_cycle(monomer_smiles, direction)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if molecule_key(molecule) == expected:
            return direction
    raise ValueError(
        "Original monomers cannot reconstruct the input macrocycle: "
        + "; ".join(errors)
    )
