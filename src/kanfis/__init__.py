"""Public API for Type-1 and interval Type-2 KANFIS."""

from .estimators import KANFISClassifier, KANFISRegressor
from .networks import IT1KANFISNetwork, IT2KANFISNetwork, build_network
from .trainer import KANFISTrainer

__version__ = "0.1.0"

__all__ = [
    "IT1KANFISNetwork",
    "IT2KANFISNetwork",
    "KANFISClassifier",
    "KANFISRegressor",
    "KANFISTrainer",
    "build_network",
]
