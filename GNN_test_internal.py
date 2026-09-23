#!/usr/bin/env python
"""Evaluate best.ckpt with one shared parent for the internal test set."""

from test_common import parse_test_args, run_test


if __name__ == "__main__":
    run_test(
        parse_test_args(
            "Evaluate CycDelta with one shared internal-test parent and export predictions."
        ),
        single_internal_parent=True,
        export_predictions=True,
    )
