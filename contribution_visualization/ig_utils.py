"""Integrated Gradients utilities for post-DMPNN node hidden states."""

from __future__ import annotations

import torch


def integrated_gradients_from_hidden(
    out_mlp: torch.nn.Module,
    child_hidden: torch.Tensor,
    parent_hidden: torch.Tensor,
    steps: int,
) -> dict:
    """Attribute final child-parent delta to child/parent nodes from zero."""

    if steps < 2:
        raise ValueError("Integrated Gradients steps must be at least 2")
    device = child_hidden.device
    if parent_hidden.device != device:
        raise ValueError("Child and parent hidden states must use the same device")

    with torch.inference_mode(False), torch.enable_grad():
        # Re-materialize inference-mode outputs as ordinary autograd tensors.
        child_actual = torch.tensor(
            child_hidden.detach().float().cpu().numpy(),
            dtype=torch.float32,
            device=device,
        )
        parent_actual = torch.tensor(
            parent_hidden.detach().float().cpu().numpy(),
            dtype=torch.float32,
            device=device,
        )
        alphas = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            dtype=torch.float32,
            device=device,
        ).view(-1, 1, 1)
        child_path = (alphas * child_actual.unsqueeze(0)).requires_grad_(True)
        parent_path = (alphas * parent_actual.unsqueeze(0)).requires_grad_(True)
        outputs = out_mlp(
            child_path.sum(dim=1) - parent_path.sum(dim=1)
        ).squeeze(-1)
        child_gradients, parent_gradients = torch.autograd.grad(
            outputs.sum(),
            (child_path, parent_path),
            create_graph=False,
            retain_graph=False,
        )

        weights = torch.ones(
            steps + 1,
            dtype=torch.float32,
            device=device,
        )
        weights[[0, -1]] = 0.5
        weights = weights.view(-1, 1, 1)
        child_average_gradient = (
            child_gradients * weights
        ).sum(dim=0) / steps
        parent_average_gradient = (
            parent_gradients * weights
        ).sum(dim=0) / steps
        child_scores = (
            child_actual * child_average_gradient
        ).sum(dim=1)
        parent_scores = (
            parent_actual * parent_average_gradient
        ).sum(dim=1)
        attribution_sum = child_scores.sum() + parent_scores.sum()
        output_difference = outputs[-1] - outputs[0]

    return {
        "child_node_scores": child_scores.detach().cpu(),
        "parent_node_scores": parent_scores.detach().cpu(),
        "baseline_output": float(outputs[0].detach().cpu()),
        "actual_output": float(outputs[-1].detach().cpu()),
        "output_difference": float(output_difference.detach().cpu()),
        "attribution_sum": float(attribution_sum.detach().cpu()),
        "completeness_error": float(
            (attribution_sum - output_difference).detach().cpu()
        ),
    }
