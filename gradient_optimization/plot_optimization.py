#!/usr/bin/env python
"""Draw aligned before/after structures for optimization suggestions."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
CONTRIBUTION_DIR = MODEL_DIR / "contribution_visualization"
for path in (MODEL_DIR, CONTRIBUTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peptide_preprocessing import parse_cyclic_peptide  # noqa: E402
from plot_contributions import orient_residues_clockwise  # noqa: E402
from chemistry import (  # noqa: E402
    assemble_validated_cycle as chemical_assemble,
    find_direction as chemical_find_direction,
    molecule_key as chemical_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将修改前和修改后的完整环肽按相同 R1/R2/...位置绘图"
    )
    parser.add_argument(
        "--result",
        default=str(SCRIPT_DIR / "optimization_result.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "optimization_plots"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="绘制前多少条建议，默认 50",
    )
    return parser.parse_args()


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
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                carboxyls.append((carbon.GetIdx(), oxygen.GetIdx()))
    nitrogens = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 7
        and not atom.GetIsAromatic()
        and not any(
            is_carbonyl(molecule, neighbor)
            for neighbor in atom.GetNeighbors()
        )
    ]
    if not carboxyls or not nitrogens:
        raise ValueError(
            "单体缺少可闭环的游离氨基或羧基: "
            + Chem.MolToSmiles(molecule, isomericSmiles=True)
        )
    choices = []
    for carbon, oxygen in carboxyls:
        for nitrogen in nitrogens:
            distance = len(
                Chem.GetShortestPath(molecule, carbon, nitrogen)
            )
            choices.append((distance, carbon, oxygen, nitrogen))
    _, carbon, oxygen, nitrogen = min(choices)
    return carbon, oxygen, nitrogen


def adjusted(index: int, removed: int) -> int:
    return index - int(index > removed)


def prepare_monomer(smiles: str) -> tuple[Chem.Mol, int, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"无法解析单体: {smiles}")
    carbon, oxygen, nitrogen = select_handles(molecule)
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(oxygen)
    return (
        editable.GetMol(),
        adjusted(carbon, oxygen),
        adjusted(nitrogen, oxygen),
    )


def assemble_cycle(
    monomer_smiles: list[str],
    direction: int,
) -> tuple[Chem.Mol, list[list[int]]]:
    monomers = [prepare_monomer(smiles) for smiles in monomer_smiles]
    combined = None
    carbons = []
    nitrogens = []
    residue_atoms = []
    offset = 0
    for molecule, carbon, nitrogen in monomers:
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
    for position, carbon in enumerate(carbons):
        target = (position + direction) % len(monomers)
        editable.AddBond(
            carbon,
            nitrogens[target],
            Chem.BondType.SINGLE,
        )
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule, residue_atoms


def molecule_key(molecule: Chem.Mol) -> str:
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def find_direction(
    input_smiles: str,
    monomer_smiles: list[str],
) -> int:
    expected = molecule_key(Chem.MolFromSmiles(input_smiles))
    for direction in (1, -1):
        rebuilt, _ = assemble_cycle(monomer_smiles, direction)
        if molecule_key(rebuilt) == expected:
            return direction
    raise ValueError(
        "原始水解单体不能准确重建输入环肽，停止作图以避免错误结构"
    )


def representative_atom(
    molecule: Chem.Mol,
    atom_ids: list[int],
) -> int:
    conformer = molecule.GetConformer()
    heavy = [
        index
        for index in atom_ids
        if molecule.GetAtomWithIdx(index).GetAtomicNum() > 1
    ] or atom_ids
    center_x = sum(
        conformer.GetAtomPosition(index).x for index in heavy
    ) / len(heavy)
    center_y = sum(
        conformer.GetAtomPosition(index).y for index in heavy
    ) / len(heavy)
    return min(
        heavy,
        key=lambda index: (
            (conformer.GetAtomPosition(index).x - center_x) ** 2
            + (conformer.GetAtomPosition(index).y - center_y) ** 2
        ),
    )


def annotate_residues(
    molecule: Chem.Mol,
    residue_atoms: list[list[int]],
) -> None:
    for position, atom_ids in enumerate(residue_atoms, start=1):
        atom = molecule.GetAtomWithIdx(
            representative_atom(molecule, atom_ids)
        )
        atom.SetProp("atomNote", f"R{position}")


def aligned_modified_depiction(
    modified: Chem.Mol,
    modified_residue_atoms: list[list[int]],
    reference: Chem.Mol,
    reference_residue_atoms: list[list[int]],
    changed_position: int,
) -> None:
    atom_map = []
    for position, (reference_ids, modified_ids) in enumerate(
        zip(reference_residue_atoms, modified_residue_atoms)
    ):
        if position == changed_position:
            continue
        if len(reference_ids) != len(modified_ids):
            raise ValueError(
                f"未修改的 R{position + 1} 原子数发生变化，不能可靠对齐"
            )
        atom_map.extend(zip(reference_ids, modified_ids))
    rdDepictor.GenerateDepictionMatching2DStructure(
        modified,
        reference,
        atom_map,
    )


def drawing_data(
    molecule: Chem.Mol,
    residue_atoms: list[list[int]],
    changed_position: int,
    color: tuple[float, float, float],
) -> tuple:
    molecule = Chem.Mol(molecule)
    annotate_residues(molecule, residue_atoms)
    changed_atoms = residue_atoms[changed_position]
    changed_set = set(changed_atoms)
    atom_colors = {index: color for index in changed_atoms}
    atom_radii = {index: 0.34 for index in changed_atoms}
    bonds = [
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if bond.GetBeginAtomIdx() in changed_set
        and bond.GetEndAtomIdx() in changed_set
    ]
    bond_colors = {index: color for index in bonds}
    return molecule, atom_colors, atom_radii, bonds, bond_colors


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_caption(drawing: bytes, suggestion: dict) -> Image.Image:
    with Image.open(BytesIO(drawing)) as source:
        source = source.convert("RGB")
        canvas = Image.new(
            "RGB",
            (source.width, source.height + 220),
            "white",
        )
        canvas.paste(source)
    draw = ImageDraw.Draw(canvas)
    center = canvas.width // 2
    y = canvas.height - 165
    draw.text(
        (center, y),
        "predicted_delta_permeability_delta = "
        f"{suggestion['predicted_permeability_delta']:.4f}",
        fill="#111827",
        font=load_font(42),
        anchor="mm",
    )
    draw.text(
        (center, y + 65),
        f"R{suggestion['position']}: "
        f"{suggestion['from_amino_acid']} -> "
        f"{suggestion['to_amino_acid']} | "
        f"{suggestion['modification_class']}",
        fill="#374151",
        font=load_font(34),
        anchor="mm",
    )
    return canvas


def draw_pair(
    suggestion: dict,
    reference: Chem.Mol,
    reference_residue_atoms: list[list[int]],
    monomers: list[str],
    direction: int,
    output: Path,
) -> str:
    changed_position = int(suggestion["position"]) - 1
    modified_monomers = list(monomers)
    modified_monomers[changed_position] = suggestion["to_fragment_smiles"]
    modified, modified_residue_atoms = chemical_assemble(
        modified_monomers,
        direction,
    )
    # Generate chemically readable bond lengths/angles first, then apply only
    # a rigid rotation/reflection convention: R1 at 12 o'clock and R2 on the
    # clockwise side. Do not stretch bonds with hard atom-coordinate locks.
    rdDepictor.Compute2DCoords(modified)
    orient_residues_clockwise(modified, modified_residue_atoms)
    before = drawing_data(
        reference,
        reference_residue_atoms,
        changed_position,
        (1.0, 0.72, 0.72),
    )
    after = drawing_data(
        modified,
        modified_residue_atoms,
        changed_position,
        (0.65, 0.90, 0.70),
    )
    max_atoms = max(before[0].GetNumAtoms(), after[0].GetNumAtoms())
    panel_width = min(1800, max(900, 650 + max_atoms * 15))
    height = max(820, int(panel_width * 0.78))
    drawer = rdMolDraw2D.MolDraw2DCairo(
        panel_width * 2,
        height,
        panel_width,
        height,
    )
    options = drawer.drawOptions()
    options.fillHighlights = True
    options.legendFontSize = 26
    options.annotationFontScale = 0.9
    drawer.DrawMolecules(
        [before[0], after[0]],
        legends=[
            f"Before | R{changed_position + 1}: "
            f"{suggestion['from_amino_acid']}",
            f"After | R{changed_position + 1}: "
            f"{suggestion['to_amino_acid']}",
        ],
        highlightAtoms=[sorted(before[1]), sorted(after[1])],
        highlightBonds=[before[3], after[3]],
        highlightAtomColors=[before[1], after[1]],
        highlightBondColors=[before[4], after[4]],
        highlightAtomRadii=[before[2], after[2]],
    )
    drawer.FinishDrawing()
    image = add_caption(drawer.GetDrawingText(), suggestion)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return chemical_key(modified)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k 必须大于 0")
    result = json.loads(
        Path(args.result).expanduser().resolve().read_text(encoding="utf-8")
    )
    suggestions = result.get("suggestions", [])
    if len(suggestions) < args.top_k:
        raise ValueError(
            f"JSON 只有 {len(suggestions)} 条建议；请先运行优化脚本生成至少 "
            f"{args.top_k} 条"
        )
    record = parse_cyclic_peptide(result["input"]["smiles"], index=0)
    monomers = [residue.smiles for residue in record.residues]
    direction = chemical_find_direction(result["input"]["smiles"], monomers)

    # Use the rebuilt molecule as the shared coordinate reference. Atom order
    # then remains known across every substitution.
    reference, reference_residue_atoms = chemical_assemble(
        monomers,
        direction,
    )
    rdDepictor.Compute2DCoords(reference)
    orient_residues_clockwise(reference, reference_residue_atoms)

    output_dir = Path(args.output_dir).expanduser().resolve()
    index_rows = []
    for rank, suggestion in enumerate(
        suggestions[: args.top_k],
        start=1,
    ):
        output = output_dir / f"optimization_rank{rank:02d}.png"
        modified_smiles = draw_pair(
            suggestion,
            reference,
            reference_residue_atoms,
            monomers,
            direction,
            output,
        )
        index_rows.append(
            {
                "rank": rank,
                "png": output.name,
                "position": suggestion["position"],
                "predicted_permeability_delta": suggestion[
                    "predicted_permeability_delta"
                ],
                "modified_smiles": modified_smiles,
            }
        )
        print(f"[{rank}/{args.top_k}] {output}")
    (output_dir / "plot_index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
