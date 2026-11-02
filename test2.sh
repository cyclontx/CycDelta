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

# Evaluate with one shared internal-test parent and export prediction CSV files.
python GNN_test_internal.py \
    -gpu_num 1 \
    -d_emb 128 \
    -n_heads 4 \
    -batch_size 32 \
    -drop_out 0.25 \
    -num_gnn_layer 2 \
    -lr 1e-4 \
    -ckpt_path ./checkpoints/best.ckpt \
    -data_root ./data \
    -seed 0 
