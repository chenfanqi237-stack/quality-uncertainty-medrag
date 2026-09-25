"""Reproducibility helpers with no heavy numerical dependencies."""

import os
import random


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

