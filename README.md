# CycDelta

CycDelta is, to our knowledge, the first model designed to predict **delta
permeability for cyclic peptides**. It directly models the permeability change
between a parent cyclic peptide and a related peptide. The model achieves strong
predictive performance while keeping the experimental input simple.

The model only needs cyclic-peptide **SMILES**, the assay **Method**, and the
permeability **label** as scientific inputs. `Index`, `Source`, and split files
are bookkeeping fields used to construct reproducible parent-child pairs.
Residue-level Uni-Mol and physicochemical features are generated locally.

## Highlights

- First delta-permeability predictor developed specifically for cyclic peptides.
- Strong predictive performance for cyclic-peptide permeability changes.
- Uni-Mol, PyTorch, RDKit, and the complete training environment are provided in
  one friendly Conda specification.
- Gradient-guided cyclic-peptide optimization provides residue-substitution
  suggestions for improving predicted permeability.

## Installation

```bash
conda env create -f env/environment_integrated.yml
conda activate permeability-gradient
```

See [env/README.md](env/README.md) for platform and GPU notes.

## Easy start: predict new SMILES without retraining

This is the recommended workflow for most users. It does **not** require the
training dataset, generated training features, or retraining. It only requires
the installed environment and the released `checkpoints/best.ckpt`.

Create a text file such as `smiles_delta_prediction/smiles.txt`:

```text
PARENT_CYCLIC_PEPTIDE_SMILES
CHILD_CYCLIC_PEPTIDE_SMILES_1
CHILD_CYCLIC_PEPTIDE_SMILES_2
```

The first non-empty line is the parent. Every following peptide is compared
with that parent. Then run:

```bash
python smiles_delta_prediction/predict.py \
  --input smiles_delta_prediction/smiles.txt \
  --output smiles_delta_prediction/predictions.csv \
  --checkpoint checkpoints/best.ckpt \
  --method PAMPA
```

The script computes the required Uni-Mol and RDKit features automatically. No
permeability label is needed for prediction. `prediction_delta` in the output
is:

```text
predicted permeability(child) - predicted permeability(parent)
```

Replace `PAMPA` with `CACO2`, `MDCK`, or `RRCK` when appropriate. If the assay
method is unavailable, use `--method unknown`; this supplies an all-zero method
encoding. See
[smiles_delta_prediction/README.md](smiles_delta_prediction/README.md) for all
options.

## Included data

```text
data/
├── CycPeptMPDB/
│   ├── cycpept_mixed_converted.csv
│   └── our_process_manifest.json
└── data_split/holdout_split/
    ├── OD_train.txt
    ├── OD_valid.txt
    ├── OD_test.txt
    ├── faris_data.txt
    ├── merz_data.xlsx
    ├── merz_data_our_process_manifest.json
    ├── Nielsen_data.xlsx
    └── nielsen_data_our_process_manifest.json
```

The main CSV contains `Index`, `SMILES`, `Method`, `Source`, and
`Permeability_Value`. The Merz table is treated as PAMPA because it does not
contain a `Method` column. Nielsen has `SMILES` and `CAPA` only; rows are
numbered from zero and evaluated with an all-zero assay vector.

No Uni-Mol tensors or RDKit descriptor files are included. Generate them
locally before training or benchmark evaluation.

## Generate features

```bash
python prepare_features.py --data-root ./data
```

Uni-Mol weights are not bundled. `unimol-tools` downloads them on first use;
if that fails, download the weights yourself. No weight path is set in this
repository.

This command:

1. validates the raw columns and writes a manifest when it is absent;
2. computes a 512-dimensional Uni-Mol representation for every residue,
   including Merz and Nielsen;
3. computes 31 RDKit physicochemical descriptors for every internal residue.

Existing files are skipped. Add `--overwrite` only when they must be replaced.
Hydrogen atoms are retained by default. Nielsen rows are numbered from zero
and use the `CAPA` column as the permeability label.

## Retrain

```bash
bash train.sh
```

or:

```bash
python GNN_train.py \
  -data_root ./data \
  -d_emb 128 \
  -n_heads 4 \
  -batch_size 32 \
  -drop_out 0.25 \
  -num_gnn_layer 2 \
  -lr 1e-4 \
  -patience 500 \
  -seed 0 \
  -max_epochs 2000
```

For each epoch, training parents are resampled with `seed + epoch`. Validation
uses ten fixed parent selections, seeds 0 through 9. Pearson correlation is
computed independently for each seed and then averaged. Early stopping and
model selection maximize `val/pearson_mean`.

Only validation metrics are computed during training. No internal-test,
Faris, or Merz metrics are inspected. The single best checkpoint is saved
as:

```text
checkpoints/best.ckpt
```

## Reproduce the reported benchmark results

Use the released `checkpoints/best.ckpt`; retraining is not required. From the
repository root:

```bash
python seed_sensitivity/seed_sensitivity.py --protocols test
```

This evaluates seeds 0 through 9. Each seed draws one parent independently for
each literature source, and features are loaded once. The reported numbers are
the mean and population standard deviation across those seeds. The datasets are:

- `internal_test`: the packaged `OD_test.txt` split;
- `faris_data`: the packaged Faris external set (`faris_data.txt`);
- `merz_data`: the external Merz dataset, scored with the PAMPA assay vector;
- `nielsen_data`: the Nielsen CAPA set, scored with an all-zero assay vector.

Summaries are written to `seed_sensitivity/results/`. Per-residue contribution
plots use `contribution_visualization/collect_node_contributions.py`, then
`gradient_optimization/plot_contributions.py`.

## Additional inference and design tools

- Gradient-guided residue optimization:
  [gradient_optimization/README.md](gradient_optimization/README.md)
- Residue contribution plots:
  `contribution_visualization/collect_node_contributions.py` and
  `gradient_optimization/plot_contributions.py`

## Main outputs

- Training metrics: `log/`
- Best checkpoint: `checkpoints/best.ckpt`
- Benchmark summaries: `seed_sensitivity/results/`

## Reproducibility notes

- Use the packaged split files without modification for reported experiments.
- Do not use test datasets for checkpoint selection.
- Keep the default Uni-Mol hydrogen setting when reproducing training features.
- `merz_data` is the canonical name for the external Merz evaluation set.
- `faris_data` is the canonical name for the external Faris evaluation set.
- `nielsen_data` uses the Nielsen CAPA labels and an all-zero assay vector.
