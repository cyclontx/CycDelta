# Permeability Prediction from SMILES

This command predicts zero-corrected permeability changes and absolute
permeability values for cyclic peptides.

## Input

Create a text file with one query SMILES per non-empty line. Line order no
longer determines the reference when a sufficiently similar training
reference is found. Otherwise the first line is used as a fallback reference.

## Default reference selection

From the repository root:

```bash
python smiles_delta_prediction/predict.py \
  --input smiles_delta_prediction/smiles.txt \
  --output smiles_delta_prediction/predictions.csv \
  --checkpoint checkpoints/best.ckpt \
  --method PAMPA
```

The default workflow:

1. uses the first input SMILES to search assay-matched `OD_train` structures;
2. selects the single most similar structure by Morgan Tanimoto similarity;
3. if similarity is at least `0.700`, uses that measured training structure as
   the common reference and outputs both delta and absolute permeability;
4. otherwise uses the first input SMILES as the common reference, outputs only
   relative permeability values, and leaves absolute-value columns empty.

When the threshold is not met, a final console warning recommends supplying a
reference SMILES and measured permeability. It is printed after Uni-Mol logs.

## User-specified reference

Use one reference for every query:

```bash
python smiles_delta_prediction/predict.py \
  --input smiles_delta_prediction/smiles.txt \
  --output smiles_delta_prediction/predictions.csv \
  --method PAMPA \
  --reference-smiles "REFERENCE_SMILES" \
  --reference-value-1e-6 1.25
```

`--reference-value-1e-6` is in units of `10^-6 cm/s`. Alternatively,
`--reference-value-log10` is `log10(Papp [cm/s])`.

A user-specified reference must include one measured permeability value.
SMILES alone cannot provide the absolute anchor for a delta model.

## Zero-point correction

The reported log delta is:

```text
out_mlp(h_query - h_reference) - out_mlp(0)
```

A reference compared with itself is therefore exactly zero. This removes the
constant output-head bias without retraining the checkpoint.

## Output

Important output columns:

- `delta_log10`: change in `log10(Papp [cm/s])`;
- `absolute_log10_cm_s`: absolute `log10(Papp [cm/s])`, when an anchor exists;
- `absolute_1e-6_cm_s`: absolute permeability in `10^-6 cm/s`.

The CSV intentionally omits similarity, threshold, status, and message
columns. Reference-selection details and warnings are reported in the console.

Supported assay methods are `PAMPA`, `CACO2`, `MDCK`, and `RRCK`.
For `method=unknown`, the default workflow falls back to the first input line
and returns delta only; absolute values require a user reference and value.

The normalization constants in `descriptor_normalizer.json` must match the
checkpoint and must not be recomputed from prediction inputs.
