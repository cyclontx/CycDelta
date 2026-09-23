# Gradient-Guided Cyclic-Peptide Optimization

This module proposes single-residue substitutions that increase the CycDelta
prediction relative to an input cyclic peptide.

## Prerequisites

Activate the integrated environment and generate the training-set features once:

```bash
conda activate permeability-gradient
python process_monomer_descriptors.py --data-root ./data
```

The optimizer only needs these generated training-residue descriptors to
recover the training-set normalization statistics. It does not need the large
precomputed training Uni-Mol tensors or precomputed features for the peptide
being optimized.

## 1. Build the full-database candidate library

```bash
python gradient_optimization/build_candidate_library.py \
  --data-root ./data \
  --candidate-scope all \
  --output gradient_optimization/candidate_library.pt
```

The default `all` scope collects unique monomers from every record in the main
database manifest, including training, validation, internal-test, and holdout
records. The complete set of 20 standard L-amino acids is also included.

The monomer library is therefore **not limited to the training split**. Some
candidate monomers may be outside the model's well-learned training
distribution, so their predicted improvements can be inaccurate. Treat all
optimization outputs as hypotheses requiring chemical review and experimental
validation. Use `--candidate-scope train` only when a train-restricted library
is specifically required.

The library builder calculates and stores fresh Uni-Mol and RDKit features for
every candidate monomer.

## 2. Optimize a peptide

```bash
python gradient_optimization/optimize_smiles.py \
  --smiles 'YOUR_CYCLIC_PEPTIDE_SMILES' \
  --checkpoint checkpoints/best.ckpt \
  --candidate-library gradient_optimization/candidate_library.pt \
  --data-root ./data \
  --method PAMPA \
  --output gradient_optimization/optimization_result.json
```

The script computes local Uni-Mol and RDKit features, differentiates the model
score with respect to residue features, screens candidate substitutions, checks
chemical validity, and ranks accepted molecules by predicted improvement.

Feature handling is therefore split into three explicit steps:

1. `process_monomer_descriptors.py` supplies training-set normalization
   statistics.
2. `build_candidate_library.py` computes candidate-monomer features once.
3. `optimize_smiles.py` recomputes features for the input peptide at runtime and
   performs a full model evaluation for every screened substitution.

For a peptide with `L` residues, the current script requires
`screen-top-k >= screen-per-position × L` and
`top-k >= min-per-position × L`. Increase `--screen-top-k` or `--top-k`, or
reduce the corresponding per-position minimum, when optimizing longer peptides.

## 3. Plot results

```bash
python gradient_optimization/plot_optimization.py \
  --result gradient_optimization/optimization_result.json
```

## Notes

- Proposed molecules are model-generated hypotheses, not experimental results.
- Keep Uni-Mol hydrogen handling consistent with training.
- The default candidate library uses the full database plus the 20 standard
  L-amino acids; inspect `candidate_scope` in the saved library before
  optimization.
- Monomers outside the training distribution can produce unreliable scores.
- Inspect chemical validity and synthetic feasibility before experimental use.
- Run each script with `--help` for device, screening, ranking, and output
  options.
