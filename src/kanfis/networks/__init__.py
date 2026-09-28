"""KANFIS network variants."""

from __future__ import annotations

from torch import nn

from .it1 import IT1KANFISNetwork, Type1KANLayer
from .it2 import IT2GaussianKANLayer, IT2KANFISNetwork


def build_network(
    variant: str,
    in_features: int,
    n_rules: int,
    out_features: int,
    n_layers: int = 1,
    n_memberships: int = 3,
    membership: str = "gaussian",
) -> nn.Module:
    """Build a Type-1 or interval Type-2 KANFIS network."""

    normalized_variant = variant.lower()
    normalized_membership = membership.lower()
    if normalized_variant == "it1":
        return IT1KANFISNetwork(
            in_features=in_features,
            n_rules=n_rules,
            out_features=out_features,
            n_layers=n_layers,
            n_memberships=n_memberships,
            membership=normalized_membership,
        )
    if normalized_variant == "it2":
        if normalized_membership != "gaussian":
            raise ValueError("IT2-KANFIS supports Gaussian membership only.")
        return IT2KANFISNetwork(
            in_features=in_features,
            n_rules=n_rules,
            out_features=out_features,
            n_layers=n_layers,
            n_memberships=n_memberships,
        )
    raise ValueError("variant must be either 'it1' or 'it2'.")


__all__ = [
    "IT1KANFISNetwork",
    "IT2GaussianKANLayer",
    "IT2KANFISNetwork",
    "Type1KANLayer",
    "build_network",
]
