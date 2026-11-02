#!/usr/bin/env python
"""Evaluate best.ckpt with per-source parent selection."""

from test_common import parse_test_args, run_test


if __name__ == "__main__":
    run_test(
        parse_test_args("Evaluate CycDelta with per-source parent selection."),
        single_internal_parent=False,
        export_predictions=False,
    )
