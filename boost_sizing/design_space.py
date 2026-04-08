from __future__ import annotations

from itertools import product
import numpy as np

from .config import DesignSpaceConfig
from .types import Design


def enumerate_designs(cfg: DesignSpaceConfig) -> list[Design]:
    return [
        Design(battery_kwh=float(b), pv_kw=float(p))
        for b, p in product(cfg.battery_sizes_kwh, cfg.pv_sizes_kw)
    ]


def sample_designs(designs: list[Design], n: int, seed: int) -> list[Design]:
    if n >= len(designs):
        return list(designs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(designs), size=n, replace=False)
    return [designs[int(i)] for i in idx]
