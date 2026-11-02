#!/usr/bin/env python
"""Record per-residue Integrated Gradients for the released CycDelta checkpoint.

The scores are computed from the post-DMPNN node states, before sum pooling.
They match the columns expected by gradient_optimization/plot_contributions.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN import QJGNN
from contribution_visualization.ig_utils import integrated_gradients_from_hidden
from data_pipeline import default_data_root
from test_common import TEST_NAMES, load_released_weights, prepare_test


def _capture_pool(model: QJGNN, calls: list[dict]) -> None:
    gnn = model.model

    def pool(graph_batch, num_graphs):
        node_feature = graph_batch["node_feature"]
        delta_unimol_feature = graph_batch["delta_unimol_feature"]
        monomer_property_feature = graph_batch["monomer_property_feature"]
        node_batch = graph_batch["node_batch"]
        hidden = gnn.node_projection(
            torch.cat(
                [
                    gnn.delta_unimol_projection(delta_unimol_feature),
                    gnn.monomer_property_projection(monomer_property_feature),
                    gnn.x_projection(node_feature),
                ],
                dim=-1,
            )
        )
        hidden = gnn.dmpnn(
            hidden,
            gnn.edge_projection(graph_batch["edge_attr"]),
            graph_batch["edge_index"],
        )
        calls.append(
            {
                "hidden": hidden.detach(),
                "node_batch": node_batch.detach(),
            }
        )
        from torch_geometric.utils import scatter

        return scatter(
            hidden,
            node_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )

    gnn._pool_graph = pool


class ContributionCallback(pl.Callback):
    def __init__(self, steps: int, calls: list[dict]):
        self.steps = steps
        self.calls = calls
        self.rows: list[dict] = []

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if len(self.calls) != 2:
            raise RuntimeError(
                f"Expected child and parent hidden states, got {len(self.calls)}"
            )
        child_call, parent_call = self.calls
        self.calls.clear()
        test_set = TEST_NAMES[dataloader_idx]
        predictions = outputs["prediction"]
        labels = outputs["label"]
        for pair_position, (child_index, parent_index, prediction, target) in enumerate(
            zip(
                outputs["index"].tolist(),
                outputs["parent_index"].tolist(),
                predictions.tolist(),
                labels.tolist(),
            )
        ):
            child_hidden = child_call["hidden"][
                child_call["node_batch"] == pair_position
            ]
            parent_hidden = parent_call["hidden"][
                parent_call["node_batch"] == pair_position
            ]
            attribution = integrated_gradients_from_hidden(
                pl_module.model.out_mlp,
                child_hidden,
                parent_hidden,
                self.steps,
            )
            if abs(attribution["actual_output"] - float(prediction)) > 1e-4:
                raise RuntimeError(
                    f"{test_set} Index={int(child_index)}: Integrated Gradients "
                    "did not reconstruct the model output."
                )
            role_data = (
                ("child", int(child_index), child_hidden, attribution["child_node_scores"]),
                ("parent", int(parent_index), parent_hidden, attribution["parent_node_scores"]),
            )
            for role, molecule_index, hidden, node_scores in role_data:
                for residue_position, (node_hidden, ig_score) in enumerate(
                    zip(hidden.detach().float().cpu(), node_scores),
                    start=1,
                ):
                    self.rows.append(
                        {
                            "test_set": test_set,
                            "batch_index": int(batch_idx),
                            "pair_position": int(pair_position),
                            "child_index": int(child_index),
                            "parent_index": int(parent_index),
                            "role": role,
                            "molecule_index": molecule_index,
                            "residue_position": residue_position,
                            "embedding_dim": int(hidden.shape[1]),
                            "ig_contribution": float(ig_score),
                            "ig_absolute": abs(float(ig_score)),
                            "raw_total": float(node_hidden.sum()),
                            "prediction_delta": float(prediction),
                            "true_delta": float(target),
                            "ig_baseline_output": attribution["baseline_output"],
                            "ig_output_difference": attribution["output_difference"],
                            "ig_attribution_sum": attribution["attribution_sum"],
                            "ig_completeness_error": attribution["completeness_error"],
                            "ig_steps": self.steps,
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "gradient_optimization" / "outputs" / "node_contributions.csv"),
    )
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "best.ckpt"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    prepared = prepare_test(
        args.data_root,
        args.batch_size,
        num_workers=0,
        single_internal_parent=False,
    )
    prepared.resample(args.seed)
    model = QJGNN(test_names=TEST_NAMES)
    load_released_weights(model, checkpoint.resolve())
    calls: list[dict] = []
    _capture_pool(model, calls)
    callback = ContributionCallback(args.steps, calls)
    trainer = pl.Trainer(
        devices=1,
        accelerator=args.accelerator,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.test(model, datamodule=prepared.datamodule, verbose=False)
    frame = pd.DataFrame(callback.rows)
    if frame.empty:
        raise RuntimeError("No node contributions were recorded.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
