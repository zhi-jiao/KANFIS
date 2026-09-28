"""Type-1 KANFIS layers and network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Type1KANLayer(nn.Module):
    """KAN layer with learnable Type-1 membership functions and rule masks."""

    VALID_MEMBERSHIPS = {"gaussian", "sigmoid", "triangle"}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_memberships: int = 3,
        membership: str = "gaussian",
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or n_memberships < 1:
            raise ValueError("Feature, rule, and membership counts must be positive.")
        membership = membership.lower()
        if membership not in self.VALID_MEMBERSHIPS:
            accepted = ", ".join(sorted(self.VALID_MEMBERSHIPS))
            raise ValueError(f"Unknown membership {membership!r}; expected one of: {accepted}.")

        self.in_features = in_features
        self.out_features = out_features
        self.n_memberships = n_memberships
        self.membership = membership

        shape = (in_features, out_features, n_memberships)
        self.centers = nn.Parameter(torch.empty(shape).uniform_(-1.0, 1.0))
        self.widths_raw = nn.Parameter(torch.full(shape, -2.0))
        self.mask = nn.Parameter(torch.ones(in_features, out_features))

    def widths(self) -> torch.Tensor:
        """Return positive membership widths (or sigmoid slopes)."""

        return F.softplus(self.widths_raw) + 1e-5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected a 2D tensor with {self.in_features} features; "
                f"received shape {tuple(x.shape)}."
            )

        expanded = x[:, :, None, None]
        widths = self.widths()

        if self.membership == "gaussian":
            membership_values = torch.exp(
                -torch.square(expanded - self.centers) / (2.0 * torch.square(widths))
            )
        elif self.membership == "sigmoid":
            membership_values = torch.sigmoid(widths * (expanded - self.centers))
        else:
            membership_values = torch.relu(
                1.0 - torch.abs(expanded - self.centers) / widths
            )

        edge_output = membership_values.sum(dim=-1)
        masked_edges = edge_output * self.mask
        hidden_output = masked_edges.sum(dim=1)
        return hidden_output, masked_edges


class IT1KANFISNetwork(nn.Module):
    """Type-1 KANFIS network whose first hidden layer represents rules."""

    variant = "it1"

    def __init__(
        self,
        in_features: int,
        n_rules: int,
        out_features: int,
        n_layers: int = 1,
        n_memberships: int = 3,
        membership: str = "gaussian",
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
        self.membership = membership.lower()

        self.layers = nn.ModuleList(
            [
                Type1KANLayer(
                    in_features,
                    n_rules,
                    n_memberships=n_memberships,
                    membership=self.membership,
                )
            ]
        )
        self.norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.norms.append(nn.LayerNorm(n_rules))
            self.layers.append(
                Type1KANLayer(
                    n_rules,
                    n_rules,
                    n_memberships=n_memberships,
                    membership=self.membership,
                )
            )
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
