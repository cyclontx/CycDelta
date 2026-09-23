#!/usr/bin/env python
"""Evaluate one CycDelta checkpoint across parent-selection seeds.

Features and the model are loaded once per protocol. Each seed only redraws
the parent peptide, which is the quantity the paper averages over.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN import QJGNN
from data_pipeline import default_data_root
from test_common import TEST_NAMES, load_released_weights, prepare_test


METRICS = ("R2", "MSE", "MAE", "pearson", "spearman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best.ckpt over parent-selection seeds."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "best.ckpt"),
    )
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-emb", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-gnn-layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accelerator", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument(
        "--protocols",
        default="test",
        help="Comma-separated protocols: test (one parent per source) and/or test2.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "seed_sensitivity" / "results"),
    )
    return parser.parse_args()


def rows_for_seed(seed: int, protocol: str, results: list[dict]) -> list[dict]:
    rows = []
    for dataset, result in zip(TEST_NAMES, results):
        for metric in METRICS:
            rows.append(
                {
                    "seed": seed,
                    "protocol": protocol,
                    "dataset": dataset,
                    "metric": metric,
                    "value": float(result[f"{dataset}/{metric}"]),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["protocol"], row["dataset"], row["metric"])
        grouped.setdefault(key, []).append(row["value"])
    summary = []
    for (protocol, dataset, metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        summary.append(
            {
                "protocol": protocol,
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean": float(array.mean()),
                "std": float(array.std(ddof=0)),
                "min": float(array.min()),
                "max": float(array.max()),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    unknown = [item for item in protocols if item not in {"test", "test2"}]
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}")

    model = QJGNN(
        d_emb=args.d_emb,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_gnn_layer=args.num_gnn_layer,
        lr=args.lr,
        test_names=TEST_NAMES,
    )
    load_released_weights(model, checkpoint)
    trainer = pl.Trainer(
        devices=args.gpu_num if args.accelerator == "gpu" else 1,
        accelerator=args.accelerator,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    rows = []
    for protocol in protocols:
        prepared = prepare_test(
            args.data_root,
            args.batch_size,
            num_workers=0,
            single_internal_parent=protocol == "test2",
        )
        for seed in range(args.seed_start, args.seed_end + 1):
            prepared.resample(seed)
            print(f"protocol={protocol} seed={seed}", flush=True)
            results = trainer.test(model, datamodule=prepared.datamodule, verbose=False)
            rows.extend(rows_for_seed(seed, protocol, results))
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    write_csv(output_dir / "metrics_by_seed.csv", rows)
    write_csv(output_dir / "sensitivity_summary.csv", summarize(rows))
    print(f"Wrote {output_dir / 'sensitivity_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
