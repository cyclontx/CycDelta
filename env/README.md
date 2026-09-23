# CycDelta Environment

The integrated Conda file installs the complete training, testing, feature
generation, inference, and optimization environment.

```bash
conda env create -f env/environment_integrated.yml
conda activate permeability-gradient
```

The environment uses Python 3.10, PyTorch 2.4.1, CUDA 11.8, PyTorch Lightning,
PyTorch Geometric, RDKit 2026.3.4, `unimol-tools` 0.1.3.post1, Matplotlib, and
Pillow. The RDKit and `unimol-tools` pins match the `unimol2` environment that
reproduces the released Uni-Mol tensors. It supports
training, testing, feature generation, SMILES prediction, gradient optimization,
and all plotting scripts without an additional model-specific installation
step.

Verify the installation:

```bash
python -c "import torch, rdkit, unimol_tools, matplotlib, PIL; print(torch.__version__, torch.cuda.is_available())"
```

Uni-Mol weights are downloaded by `unimol-tools` the first time features are
generated. No weight path is configured in this environment.

For a CPU smoke test, pass `-accelerator cpu` to the training or test script.
Full training is intended for a CUDA-capable GPU.
