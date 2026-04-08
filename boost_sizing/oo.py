from __future__ import annotations

from math import comb, ceil, log


def compute_n(probability_top_alpha_hit: float, alpha_fraction: float) -> int:
    if not (0.0 < probability_top_alpha_hit < 1.0):
        raise ValueError("probability_top_alpha_hit must be in (0, 1)")
    if not (0.0 < alpha_fraction < 1.0):
        raise ValueError("alpha_fraction must be in (0, 1)")
    return ceil(log(1.0 - probability_top_alpha_hit) / log(1.0 - alpha_fraction))


def alignment_probability(k: int, g: int, s: int, n: int) -> float:
    if any(v <= 0 for v in (k, g, s, n)):
        raise ValueError("k, g, s, n must be positive")
    if g > n or s > n or k > min(g, s):
        return 0.0
    denom = comb(n, s)
    num = 0
    for i in range(k, min(g, s) + 1):
        num += comb(g, i) * comb(n - g, s - i)
    return num / denom


def choose_s(k: int, g: int, n: int, target_alignment_probability: float) -> int:
    if not (0.0 < target_alignment_probability < 1.0):
        raise ValueError("target_alignment_probability must be in (0, 1)")
    for s in range(k, n + 1):
        if alignment_probability(k, g, s, n) >= target_alignment_probability:
            return s
    return n
