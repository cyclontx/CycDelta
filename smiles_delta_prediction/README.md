# Delta-Permeability Prediction from SMILES

This utility predicts permeability changes between a parent cyclic peptide and
one or more related peptides directly from SMILES.

## Input

Create `smiles.txt` with one cyclic-peptide SMILES per non-empty line. The first
line is the parent; every later line is compared with that parent.

## Run

From the repository root:

```bash
python smiles_delta_prediction/predict.py \
  --input smiles_delta_prediction/smiles.txt \
  --output smiles_delta_prediction/predictions.csv \
  --checkpoint checkpoints/best.ckpt \
  --method PAMPA
```

Supported methods are `PAMPA`, `CACO2`, `MDCK`, `RRCK`, and `unknown`. Use
`unknown` when the assay method is unavailable; it produces an all-zero method
encoding rather than selecting one of the four trained assay categories. Use
`--device cpu` when no CUDA device is available.

The script parses each cyclic peptide, computes Uni-Mol residue
representations and RDKit descriptors locally, constructs the paired graphs,
and writes the predicted child-minus-parent permeability change.

## Output

`predictions.csv` includes the input line, normalized sequence, parent line,
assay method, predicted delta, processing status, and any error message.

The descriptor normalization constants are stored in
`descriptor_normalizer.json`. They must match the checkpoint and should not be
recomputed from prediction inputs.
