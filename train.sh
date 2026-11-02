#!/bin/bash

# Report the available accelerator.
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Train. Validation is averaged over fixed parent-selection seeds 0-9.
python GNN_train.py \
    -gpu_num 1 \
    -d_emb 128 \
    -n_heads 4 \
    -batch_size 32 \
    -drop_out 0.25 \
    -num_gnn_layer 2 \
    -lr 1e-4 \
    -data_root ./data \
    -patience 500 \
    -seed 0 \
    -max_epochs 2000
