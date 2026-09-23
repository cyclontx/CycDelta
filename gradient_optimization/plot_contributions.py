"""Draw residue-level cyclic peptide graphs annotated with pooling contributions."""

from __future__ import annotations

import argparse
import html
from io import BytesIO
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from peptide_preprocessing import _component_order, find_macrocyclic_amide_bonds


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 node_contributions.csv 绘制逐残基贡献环肽图。"
    )
    parser.add_argument(
        "--contributions",
        default=str(SCRIPT_DIR / "outputs" / "node_contributions.csv"),
        help="collect_node_contributions.py 生成的 CSV。",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="包含 CycPeptMPDB 和 data_split 的 permeability 数据根目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "outputs" / "plots"),
        help="图片输出目录。",
    )
    parser.add_argument(
        "--test-sets",
        nargs="*",
        choices=(
            "internal_test",
            "faris_data",
            "merz_data",
            "nielsen_data",
            "holdout2",
            "holdout3",
        ),
        help="只绘制指定测试集；不提供时绘制全部测试集。",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=("png",),
        help="图片格式，可同时输出多种。",
    )
    parser.add_argument(
        "--plot-style",
        choices=("molecule",),
        default="molecule",
        help="绘制 Child/Parent 并排的真实 2D 分子。",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到预处理 manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["index"]): record for record in payload["records"]}


def load_records(data_root: Path) -> dict[str, dict[int, dict]]:
    main_records = read_manifest(
        data_root / "CycPeptMPDB" / "our_process_manifest.json"
    )
    split_dir = data_root / "data_split" / "holdout_split"
    records = {
        "internal_test": main_records,
        "faris_data": main_records,
    }
    optional = {
        "merz_data": split_dir / "merz_data_our_process_manifest.json",
        "nielsen_data": split_dir / "nielsen_data_our_process_manifest.json",
        "holdout2": split_dir / "Holdout2_our_process_manifest.json",
        "holdout3": split_dir / "Holdout3_our_process_manifest.json",
    }
    for name, path in optional.items():
        if path.exists():
            records[name] = read_manifest(path)
    return records


def residue_label(residue: dict, fallback_position: int) -> str:
    amino_acid = str(residue.get("amino_acid", "X"))
    modifiers = []
    if bool(residue.get("is_D", False)):
        modifiers.append("D")
    if bool(residue.get("is_methylated", False)):
        modifiers.append("NMe")
    prefix = "-".join(modifiers)
    name = f"{prefix}-{amino_acid}" if prefix else amino_acid
    return f"{int(residue.get('position', fallback_position))}: {name}"


def structural_edges(num_residues: int, cyclized_pair) -> tuple[list, tuple]:
    backbone = [(index, index + 1) for index in range(num_residues - 1)]
    first, second = (int(value) for value in cyclized_pair)
    if not 0 <= first < num_residues or not 0 <= second < num_residues:
        raise ValueError(
            f"环化位置 {cyclized_pair} 超出 {num_residues} 个残基的范围。"
        )
    return backbone, (first, second)


def draw_edge(axis, positions, edge, **kwargs) -> None:
    first, second = edge
    axis.plot(
        [positions[first, 0], positions[second, 0]],
        [positions[first, 1], positions[second, 1]],
        **kwargs,
    )


def ordered_residue_atom_ids(record: dict) -> tuple[Chem.Mol, list[list[int]]]:
    """Reproduce preprocessing order and map each residue to original atoms."""

    molecule = Chem.MolFromSmiles(record["input_smiles"])
    if molecule is None:
        raise ValueError(f"Index={record['index']}: 无法解析 input_smiles。")

    cuts = find_macrocyclic_amide_bonds(molecule)
    editable = Chem.RWMol(molecule)
    for _, carbon_index, nitrogen_index in cuts:
        editable.RemoveBond(carbon_index, nitrogen_index)
        oxygen_index = editable.AddAtom(Chem.Atom(8))
        editable.AddBond(carbon_index, oxygen_index, Chem.BondType.SINGLE)

    hydrolysed = editable.GetMol()
    Chem.SanitizeMol(hydrolysed)
    fragment_atom_ids: list[tuple[int, ...]] = []
    fragments = list(
        Chem.GetMolFrags(
            hydrolysed,
            asMols=True,
            sanitizeFrags=True,
            fragsMolAtomMapping=fragment_atom_ids,
        )
    )
    if len(fragments) != len(cuts):
        raise ValueError(
            f"Index={record['index']}: 切分得到 {len(fragments)} 个片段，"
            f"但检测到 {len(cuts)} 个宏环酰胺键。"
        )

    original_atom_count = molecule.GetNumAtoms()
    atom_to_component = {}
    original_ids_by_component: list[list[int]] = []
    for component, atom_ids in enumerate(fragment_atom_ids):
        original_ids = [
            int(atom_index)
            for atom_index in atom_ids
            if atom_index < original_atom_count
        ]
        original_ids_by_component.append(original_ids)
        for atom_index in original_ids:
            atom_to_component[atom_index] = component

    order, used_undirected_fallback = _component_order(atom_to_component, cuts)
    ordered_fragments = [fragments[component] for component in order]
    ordered_atom_ids = [original_ids_by_component[component] for component in order]

    # This is intentionally identical to hydrolyse_macrocycle(): the manifest
    # residue positions use this canonical rotation (and possible reversal).
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
        ordered_fragments.reverse()
        ordered_atom_ids.reverse()
    ordered_fragments = (
        ordered_fragments[rotation:] + ordered_fragments[:rotation]
    )
    ordered_atom_ids = ordered_atom_ids[rotation:] + ordered_atom_ids[:rotation]

    expected_smiles = [residue["smiles"] for residue in record["residues"]]
    observed_smiles = [
        Chem.MolToSmiles(fragment, isomericSmiles=True)
        for fragment in ordered_fragments
    ]
    if observed_smiles != expected_smiles:
        mismatch = next(
            (
                position
                for position, (observed, expected) in enumerate(
                    zip(observed_smiles, expected_smiles), start=1
                )
                if observed != expected
            ),
            1,
        )
        raise ValueError(
            f"Index={record['index']}: 重新切分结果与 manifest 的第 {mismatch} "
            "个残基不一致，已停止作图以避免贡献标错原子。"
        )
    return molecule, ordered_atom_ids


def contribution_colors(
    contributions: np.ndarray,
    color_max: float,
) -> list[tuple[float, float, float]]:
    normalization = Normalize(vmin=-color_max, vmax=color_max)
    color_map = colormaps["RdBu_r"]
    colors = []
    for value in contributions:
        base = color_map(normalization(value))[:3]
        # Blend strongly towards white so atom labels and bond orders stay clear.
        colors.append(
            tuple(float(0.45 * channel + 0.55) for channel in base)
        )
    return colors


def orient_residues_clockwise(
    molecule: Chem.Mol,
    residue_atom_ids: list[list[int]],
) -> None:
    """Put the R1 centroid at 12 o'clock and R2 on its clockwise side."""

    conformer = molecule.GetConformer()

    def centroid(atom_ids: list[int]) -> np.ndarray:
        coordinates = np.array(
            [
                [
                    conformer.GetAtomPosition(atom_index).x,
                    conformer.GetAtomPosition(atom_index).y,
                ]
                for atom_index in atom_ids
            ],
            dtype=float,
        )
        return coordinates.mean(axis=0)

    all_atom_ids = [
        atom_index
        for residue_atom_id_list in residue_atom_ids
        for atom_index in residue_atom_id_list
    ]
    molecular_center = centroid(all_atom_ids)
    first_center = centroid(residue_atom_ids[0])
    first_vector = first_center - molecular_center
    if np.linalg.norm(first_vector) < 1e-8:
        raise ValueError("R1 的几何中心与分子中心重合，无法确定 12 点方向。")

    current_angle = np.arctan2(first_vector[1], first_vector[0])
    rotation_angle = np.pi / 2 - current_angle
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])

    transformed = {}
    for atom_index in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        coordinate = np.array([position.x, position.y]) - molecular_center
        transformed[atom_index] = rotation @ coordinate

    if len(residue_atom_ids) > 1:
        second_center = np.mean(
            [transformed[atom_index] for atom_index in residue_atom_ids[1]],
            axis=0,
        )
        # At 12 o'clock, clockwise progression goes towards positive x.
        if second_center[0] < 0:
            for coordinate in transformed.values():
                coordinate[0] *= -1

    for atom_index, coordinate in transformed.items():
        conformer.SetAtomPosition(
            atom_index,
            (float(coordinate[0]), float(coordinate[1]), 0.0),
        )


def prepare_annotated_molecule(
    molecule: Chem.Mol,
    residue_atom_ids: list[list[int]],
    color_values: np.ndarray,
    color_max: float,
    annotation_values: np.ndarray | None = None,
) -> tuple[Chem.Mol, dict, dict, dict, dict]:
    molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(molecule)
    orient_residues_clockwise(molecule, residue_atom_ids)
    conformer = molecule.GetConformer()
    colors = contribution_colors(color_values, color_max)
    if annotation_values is None:
        annotation_values = color_values

    atom_colors = {}
    atom_radii = {}
    atom_to_residue = {}
    for residue_index, atom_ids in enumerate(residue_atom_ids):
        for atom_index in atom_ids:
            atom_colors[atom_index] = colors[residue_index]
            atom_radii[atom_index] = 0.32
            atom_to_residue[atom_index] = residue_index

        heavy_atom_ids = [
            atom_index
            for atom_index in atom_ids
            if molecule.GetAtomWithIdx(atom_index).GetAtomicNum() > 1
        ]
        if not heavy_atom_ids:
            heavy_atom_ids = atom_ids
        coordinates = np.array(
            [
                [
                    conformer.GetAtomPosition(atom_index).x,
                    conformer.GetAtomPosition(atom_index).y,
                ]
                for atom_index in heavy_atom_ids
            ]
        )
        centroid = coordinates.mean(axis=0)
        representative_offset = int(
            np.square(coordinates - centroid).sum(axis=1).argmin()
        )
        representative_atom = heavy_atom_ids[representative_offset]
        molecule.GetAtomWithIdx(representative_atom).SetProp(
            "atomNote",
            f"R{residue_index + 1}: {annotation_values[residue_index]:.2f}",
        )

    bond_colors = {}
    highlighted_bonds = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (
            begin in atom_to_residue
            and end in atom_to_residue
            and atom_to_residue[begin] == atom_to_residue[end]
        ):
            bond_index = bond.GetIdx()
            highlighted_bonds.append(bond_index)
            bond_colors[bond_index] = colors[atom_to_residue[begin]]

    return (
        molecule,
        atom_colors,
        atom_radii,
        bond_colors,
        highlighted_bonds,
    )


def create_molecule_drawer(
    image_format: str,
    width: int,
    height: int,
    panel_width: int = -1,
    panel_height: int = -1,
):
    if image_format == "svg":
        return rdMolDraw2D.MolDraw2DSVG(
            width,
            height,
            panel_width,
            panel_height,
        )
    return rdMolDraw2D.MolDraw2DCairo(
        width,
        height,
        panel_width,
        panel_height,
    )


def caption_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_raster_captions(
    drawing: bytes,
    captions: tuple[str, str],
    caption_height: int = 290,
) -> Image.Image:
    with Image.open(BytesIO(drawing)) as source:
        source = source.convert("RGB")
        canvas = Image.new(
            "RGB",
            (source.width, source.height + caption_height),
            "white",
        )
        canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = caption_font(40)
    center_x = canvas.width // 2
    first_y = canvas.height - caption_height + 55
    draw.text(
        (center_x, first_y),
        captions[0],
        fill="#111827",
        font=font,
        anchor="mm",
    )
    draw.text(
        (center_x, first_y + 60),
        captions[1],
        fill="#111827",
        font=font,
        anchor="mm",
    )
    legend_width = min(760, canvas.width - 240)
    legend_height = 28
    legend_left = center_x - legend_width // 2
    legend_top = first_y + 115
    for offset in range(legend_width):
        fraction = offset / max(legend_width - 1, 1)
        color = tuple(
            int(round(channel * 255))
            for channel in colormaps["RdBu_r"](fraction)[:3]
        )
        draw.line(
            (
                legend_left + offset,
                legend_top,
                legend_left + offset,
                legend_top + legend_height,
            ),
            fill=color,
        )
    legend_font = caption_font(30)
    label_y = legend_top + legend_height + 28
    draw.text(
        (legend_left, label_y),
        "More hydrophilic",
        fill="#111827",
        font=legend_font,
        anchor="lm",
    )
    draw.text(
        (legend_left + legend_width, label_y),
        "More lipophilic",
        fill="#111827",
        font=legend_font,
        anchor="rm",
    )
    return canvas


def add_svg_captions(
    drawing: str,
    width: int,
    height: int,
    captions: tuple[str, str],
    caption_height: int = 290,
) -> str:
    total_height = height + caption_height
    drawing = re.sub(
        r"(height=['\"])\d+(?:\.\d+)?px",
        rf"\g<1>{total_height}px",
        drawing,
        count=1,
    )
    drawing = re.sub(
        r"(viewBox=['\"]0 0 \d+(?:\.\d+)?) \d+(?:\.\d+)?",
        rf"\g<1> {total_height}",
        drawing,
        count=1,
    )
    center_x = width / 2
    legend_width = min(760, width - 240)
    legend_left = center_x - legend_width / 2
    legend_top = height + 175
    caption_markup = (
        f"<rect x='0' y='{height}' width='{width}' height='{caption_height}' "
        "fill='#ffffff'/>\n"
        f"<text x='{center_x}' y='{height + 65}' text-anchor='middle' "
        "font-family='sans-serif' font-size='40px' fill='#111827'>"
        f"{html.escape(captions[0])}</text>\n"
        f"<text x='{center_x}' y='{height + 125}' text-anchor='middle' "
        "font-family='sans-serif' font-size='40px' fill='#111827'>"
        f"{html.escape(captions[1])}</text>\n"
        "<defs><linearGradient id='hydrophilicityLegend' x1='0%' y1='0%' "
        "x2='100%' y2='0%'>"
        "<stop offset='0%' stop-color='#2166ac'/>"
        "<stop offset='50%' stop-color='#f7f7f7'/>"
        "<stop offset='100%' stop-color='#b2182b'/>"
        "</linearGradient></defs>\n"
        f"<rect x='{legend_left}' y='{legend_top}' width='{legend_width}' "
        "height='28' fill='url(#hydrophilicityLegend)'/>\n"
        f"<text x='{legend_left}' y='{legend_top + 62}' text-anchor='start' "
        "font-family='sans-serif' font-size='30px' fill='#111827'>"
        "More hydrophilic</text>\n"
        f"<text x='{legend_left + legend_width}' y='{legend_top + 62}' "
        "text-anchor='end' font-family='sans-serif' font-size='30px' "
        "fill='#111827'>More lipophilic</text>\n"
    )
    return drawing.replace("</svg>", f"{caption_markup}</svg>")


def draw_2d_molecule(
    dataframe: pd.DataFrame,
    record: dict,
    test_set: str,
    role: str,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    dataframe = dataframe.sort_values("residue_position")
    residues = record["residues"]
    if len(dataframe) != len(residues):
        raise ValueError(
            f"{test_set} Index={record['index']}: CSV 有 {len(dataframe)} 个节点，"
            f"manifest 有 {len(residues)} 个残基。"
        )
    contributions = dataframe["node_sum"].to_numpy(dtype=float)
    molecule, residue_atom_ids = ordered_residue_atom_ids(record)
    (
        annotated_molecule,
        atom_colors,
        atom_radii,
        bond_colors,
        highlighted_bonds,
    ) = prepare_annotated_molecule(
        molecule,
        residue_atom_ids,
        contributions,
        max(float(np.nanmax(contributions)), 1e-9),
    )

    width = min(2400, max(1100, 700 + molecule.GetNumAtoms() * 18))
    height = max(850, int(width * 0.72))
    legend = (
        f"{test_set} | {role} | Index={record['index']} | "
        f"raw sum={contributions.sum():.2f}"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        drawer_format = "png" if image_format == "pdf" else image_format
        drawer = create_molecule_drawer(drawer_format, width, height)
        options = drawer.drawOptions()
        options.fillHighlights = True
        options.legendFontSize = 24
        options.annotationFontScale = 0.85
        drawer.DrawMolecule(
            annotated_molecule,
            legend=legend,
            highlightAtoms=sorted(atom_colors),
            highlightBonds=highlighted_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
            highlightAtomRadii=atom_radii,
        )
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
        output_path = output_stem.with_suffix(f".{image_format}")
        if image_format == "svg":
            output_path.write_text(drawing, encoding="utf-8")
        elif image_format == "png":
            output_path.write_bytes(drawing)
        else:
            with Image.open(BytesIO(drawing)) as image:
                image.convert("RGB").save(
                    output_path,
                    format="PDF",
                    resolution=dpi,
                )


def draw_2d_pair(
    child_dataframe: pd.DataFrame,
    parent_dataframe: pd.DataFrame,
    child_record: dict,
    parent_record: dict,
    prediction_delta: float,
    true_delta: float,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    child_dataframe = child_dataframe.sort_values("residue_position")
    parent_dataframe = parent_dataframe.sort_values("residue_position")
    child_values = child_dataframe["ig_contribution"].to_numpy(dtype=float)
    parent_values = parent_dataframe["ig_contribution"].to_numpy(dtype=float)
    if len(child_values) != len(child_record["residues"]):
        raise ValueError(
            f"Child Index={child_record['index']}: 节点数与 manifest 不一致。"
        )
    if len(parent_values) != len(parent_record["residues"]):
        raise ValueError(
            f"Parent Index={parent_record['index']}: 节点数与 manifest 不一致。"
        )

    color_max = max(
        float(np.nanmax(np.abs(child_values))),
        float(np.nanmax(np.abs(parent_values))),
        1e-9,
    )
    # Parent enters the model as child_embedding - parent_embedding. Reverse
    # only its display colors so both panels use the same molecule-intrinsic
    # direction; retain the original signed IG values in residue annotations.
    prepared = []
    for record, color_values, annotation_values in (
        (parent_record, -parent_values, parent_values),
        (child_record, child_values, child_values),
    ):
        molecule, residue_atom_ids = ordered_residue_atom_ids(record)
        annotated = prepare_annotated_molecule(
            molecule,
            residue_atom_ids,
            color_values,
            color_max,
            annotation_values,
        )
        prepared.append(annotated)

    molecules = [item[0] for item in prepared]
    atom_colors = [item[1] for item in prepared]
    atom_radii = [item[2] for item in prepared]
    bond_colors = [item[3] for item in prepared]
    highlighted_bonds = [item[4] for item in prepared]
    max_atoms = max(molecule.GetNumAtoms() for molecule in molecules)
    panel_width = min(1900, max(950, 650 + max_atoms * 15))
    height = max(850, int(panel_width * 0.78))
    legends = ["Parent", "Child"]
    change_captions = (
        f"Predicted change relative to Parent: {prediction_delta:.4f}",
        f"True change relative to Parent: {true_delta:.4f}",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        drawer_format = "png" if image_format == "pdf" else image_format
        drawer = create_molecule_drawer(
            drawer_format,
            panel_width * 2,
            height,
            panel_width,
            height,
        )
        options = drawer.drawOptions()
        options.fillHighlights = True
        options.legendFontSize = 22
        options.annotationFontScale = 0.82
        drawer.DrawMolecules(
            molecules,
            legends=legends,
            highlightAtoms=[sorted(colors) for colors in atom_colors],
            highlightBonds=highlighted_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
            highlightAtomRadii=atom_radii,
        )
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
        output_path = output_stem.with_suffix(f".{image_format}")
        if image_format == "svg":
            output_path.write_text(
                add_svg_captions(
                    drawing,
                    panel_width * 2,
                    height,
                    change_captions,
                ),
                encoding="utf-8",
            )
        else:
            image = add_raster_captions(
                drawing,
                change_captions,
            )
            if image_format == "png":
                image.save(output_path, format="PNG")
            else:
                image.save(
                    output_path,
                    format="PDF",
                    resolution=dpi,
                )


def draw_peptide(
    dataframe: pd.DataFrame,
    record: dict,
    test_set: str,
    role: str,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    dataframe = dataframe.sort_values("residue_position")
    residues = record["residues"]
    if len(dataframe) != len(residues):
        raise ValueError(
            f"{test_set} Index={record['index']}: CSV 有 {len(dataframe)} 个节点，"
            f"manifest 有 {len(residues)} 个残基。"
        )

    expected_positions = list(range(1, len(residues) + 1))
    actual_positions = dataframe["residue_position"].astype(int).tolist()
    if actual_positions != expected_positions:
        raise ValueError(
            f"{test_set} Index={record['index']}: 残基位置不连续: {actual_positions}"
        )

    contributions = dataframe["contribution_percent"].to_numpy(dtype=float)
    num_residues = len(residues)
    angles = np.pi / 2 - 2 * np.pi * np.arange(num_residues) / num_residues
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    backbone, cyclization = structural_edges(
        num_residues, record["cyclized_pair"]
    )

    figure_size = max(7.5, min(13.0, 6.5 + num_residues * 0.22))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    for edge in backbone:
        draw_edge(
            axis,
            positions,
            edge,
            color="#6b7280",
            linewidth=2.4,
            zorder=1,
        )
    draw_edge(
        axis,
        positions,
        cyclization,
        color="#2563eb",
        linewidth=3.0,
        linestyle="--",
        zorder=2,
    )

    color_max = max(float(np.nanmax(contributions)), 1e-9)
    normalization = Normalize(vmin=0.0, vmax=color_max)
    nodes = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=contributions,
        cmap="YlOrRd",
        norm=normalization,
        s=max(1050, 1750 - num_residues * 25),
        edgecolors="#111827",
        linewidths=1.5,
        zorder=3,
    )

    for index, (x_coordinate, y_coordinate) in enumerate(positions):
        relative_intensity = contributions[index] / color_max
        axis.text(
            x_coordinate,
            y_coordinate,
            f"{index + 1}\n{contributions[index]:.1f}%",
            ha="center",
            va="center",
            fontsize=max(7.0, 9.8 - num_residues * 0.10),
            fontweight="bold",
            color="white" if relative_intensity >= 0.55 else "#111827",
            zorder=4,
        )
        label_radius = 1.25
        axis.text(
            label_radius * x_coordinate,
            label_radius * y_coordinate,
            residue_label(residues[index], index + 1),
            ha="center",
            va="center",
            fontsize=max(7.0, 10.5 - num_residues * 0.12),
            color="#111827",
        )

    total = contributions.sum()
    sequence = str(record.get("sequence", ""))
    axis.set_title(
        f"{test_set} | {role} | Index={record['index']}\n"
        f"Sequence: {sequence} | pooling contribution sum={total:.2f}%",
        fontsize=13,
        pad=20,
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color="#6b7280", lw=2.4, label="Backbone"),
            Line2D(
                [0],
                [0],
                color="#2563eb",
                lw=3.0,
                linestyle="--",
                label=(
                    f"Cyclization: {cyclization[0] + 1}"
                    f"–{cyclization[1] + 1}"
                ),
            ),
        ],
        loc="lower center",
        frameon=False,
        ncol=2,
    )
    colorbar = figure.colorbar(nodes, ax=axis, shrink=0.70, pad=0.02)
    colorbar.set_label("Normalized pooling contribution (%)")
    axis.set_aspect("equal")
    axis.set_xlim(-1.55, 1.55)
    axis.set_ylim(-1.50, 1.50)
    axis.axis("off")
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for image_format in formats:
        figure.savefig(
            output_stem.with_suffix(f".{image_format}"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def main() -> None:
    args = parse_arguments()
    contribution_path = Path(args.contributions).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    dataframe = pd.read_csv(contribution_path)
    required_columns = {
        "test_set",
        "batch_index",
        "pair_position",
        "child_index",
        "parent_index",
        "role",
        "molecule_index",
        "residue_position",
        "ig_contribution",
        "raw_total",
        "prediction_delta",
        "true_delta",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"贡献 CSV 缺少列: {sorted(missing)}")

    if args.test_sets:
        dataframe = dataframe[dataframe["test_set"].isin(args.test_sets)].copy()
    if dataframe.empty:
        raise ValueError("按当前 --test-sets 筛选后没有可绘制记录。")

    records_by_test_set = load_records(data_root)
    plot_rows = []
    group_columns = [
        "test_set",
        "batch_index",
        "pair_position",
        "child_index",
        "parent_index",
    ]
    for pair_key, group in dataframe.groupby(group_columns, sort=True):
        test_set, batch_index, pair_position, child_index, parent_index = pair_key
        child_group = group[group["role"] == "child"].copy()
        parent_group = group[group["role"] == "parent"].copy()
        if child_group.empty or parent_group.empty:
            raise ValueError(
                f"{test_set} batch={batch_index} pair={pair_position}: "
                "缺少 child 或 parent 节点记录。"
            )
        prediction_values = group["prediction_delta"].drop_duplicates()
        true_values = group["true_delta"].drop_duplicates()
        if len(prediction_values) != 1 or len(true_values) != 1:
            raise ValueError(
                f"{test_set} batch={batch_index} pair={pair_position}: "
                "预测差值或真实差值记录不唯一。"
            )
        prediction_delta = float(prediction_values.iloc[0])
        true_delta = float(true_values.iloc[0])

        child_index = int(child_index)
        parent_index = int(parent_index)
        child_record = records_by_test_set[test_set].get(child_index)
        parent_record = records_by_test_set[test_set].get(parent_index)
        if child_record is None:
            raise KeyError(
                f"{test_set} manifest 中找不到 Child Index={child_index}。"
            )
        if parent_record is None:
            raise KeyError(
                f"{test_set} manifest 中找不到 Parent Index={parent_index}。"
            )
        stem = (
            output_dir
            / safe_filename(test_set)
            / (
                f"batch_{int(batch_index):04d}_pair_{int(pair_position):03d}"
                f"_child_{child_index}_parent_{parent_index}"
            )
        )
        draw_2d_pair(
            child_group,
            parent_group,
            child_record,
            parent_record,
            prediction_delta,
            true_delta,
            stem,
            tuple(args.formats),
            args.dpi,
        )
        plot_rows.append(
            {
                "test_set": test_set,
                "batch_index": int(batch_index),
                "pair_position": int(pair_position),
                "child_index": child_index,
                "parent_index": parent_index,
                "child_num_residues": len(child_record["residues"]),
                "parent_num_residues": len(parent_record["residues"]),
                "prediction_delta": prediction_delta,
                "true_delta": true_delta,
                "plot_style": args.plot_style,
                "output_stem": str(stem),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(output_dir / "plot_index.csv", index=False)
    print(
        f"已绘制 {len(plot_rows)} 对 Child/Parent，图片目录: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
