"""Interval Type-2 KANFIS layers and network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class IT2GaussianKANLayer(nn.Module):
    """Gaussian interval Type-2 KAN layer with learnable rule masks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_memberships: int = 3,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or n_memberships < 1:
            raise ValueError("Feature, rule, and membership counts must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.n_memberships = n_memberships

        shape = (in_features, out_features, n_memberships)
        self.centers = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.lower_widths_raw = nn.Parameter(torch.full(shape, -2.0))
        self.width_differences_raw = nn.Parameter(torch.full(shape, -2.0))
        self.mask = nn.Parameter(torch.ones(in_features, out_features))

    def widths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered positive lower and upper Gaussian widths."""

        lower = F.softplus(self.lower_widths_raw) + 1e-5
        upper = lower + F.softplus(self.width_differences_raw) + 1e-5
        return lower, upper

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected a 2D tensor with {self.in_features} features; "
                f"received shape {tuple(x.shape)}."
            )

        expanded = x[:, :, None, None]
        lower_widths, upper_widths = self.widths()
        squared_distance = torch.square(expanded - self.centers)
        lower_membership = torch.exp(
            -squared_distance / (2.0 * torch.square(lower_widths))
        )
        upper_membership = torch.exp(
            -squared_distance / (2.0 * torch.square(upper_widths))
        )

        interval_midpoint = (lower_membership + upper_membership) / 2.0
        edge_output = interval_midpoint.sum(dim=-1)
        masked_edges = edge_output * self.mask
        hidden_output = masked_edges.sum(dim=1)
        return hidden_output, masked_edges


class IT2KANFISNetwork(nn.Module):
    """Interval Type-2 KANFIS network with Gaussian uncertainty intervals."""

    variant = "it2"
    membership = "gaussian"

    def __init__(
        self,
        in_features: int,
        n_rules: int,
        out_features: int,
        n_layers: int = 1,
        n_memberships: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if out_features < 1:
            raise ValueError("out_features must be positive.")

        self.in_features = in_features
        self.n_rules = n_rules
        self.out_features = out_features
        self.n_layers = n_layers
        self.n_memberships = n_memberships

        self.layers = nn.ModuleList(
            [IT2GaussianKANLayer(in_features, n_rules, n_memberships)]
        )
        self.norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.norms.append(nn.LayerNorm(n_rules))
            self.layers.append(IT2GaussianKANLayer(n_rules, n_rules, n_memberships))
        self.final_layer = nn.Linear(n_rules, out_features)

    def get_mask(self) -> torch.Tensor:
        """Return the first-layer rule-by-feature mask."""

        return self.layers[0].mask.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = x
        first_layer_firing: torch.Tensor | None = None
        for index, layer in enumerate(self.layers):
            current, _ = layer(current)
            if index == 0:
                first_layer_firing = current
            if index < self.n_layers - 1:
                current = self.norms[index](current)

        prediction = self.final_layer(current)
        if first_layer_firing is None:
            raise RuntimeError("KANFIS network has no rule layer.")
        return prediction, first_layer_firing
