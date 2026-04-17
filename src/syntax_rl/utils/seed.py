"""Reproducible seeding helpers."""

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """Record of seed values applied to supported libraries."""

    seed: int
    python_hash_seed: str


def seed_everything(seed: int) -> SeedState:
    """Seed Python, NumPy, and hash behavior for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return SeedState(seed=seed, python_hash_seed=os.environ["PYTHONHASHSEED"])
