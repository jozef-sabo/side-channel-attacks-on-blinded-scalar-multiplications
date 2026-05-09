from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackParams:
    K: int
    E: int
    eps_b: float
    eps_b_compl: float
    eps_b_multipliers: list[float]
    eps_b_compl_multipliers: list[float]


def _mask(width: int) -> int:
    return (1 << width) - 1


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def evaluate_probability(
        r_window_pred: int,
        d_window_pred: int,
        noisy_scalar: int,
        start_bit: int,
        window_size: int,
        params: AttackParams,
) -> float:
    window_mask = _mask(window_size)

    noisy_r_window = (noisy_scalar >> (params.K + start_bit)) & window_mask
    noisy_d_window = (noisy_scalar >> start_bit) & window_mask

    h = hamming_distance(r_window_pred, noisy_r_window) + hamming_distance(d_window_pred, noisy_d_window)
    return params.eps_b_multipliers[h] * params.eps_b_compl_multipliers[2 * window_size - h]


def improved_algorithm_step1(
        iteration: int,
        window_t: int,
        noisy_scalars: list[int],
        Lr: list[list[tuple[float, int]]],
        d_mod_2_i_minus_1: int,
        params: AttackParams,
) -> int:
    scores = [0.0, 0.0]

    width = min(window_t, iteration)
    start_bit = max(iteration - width, 0)
    d_low = d_mod_2_i_minus_1 & _mask(iteration - 1)
    mod_i = _mask(iteration)

    for d_hat in (0, 1):
        d_bar = d_low | (d_hat << (iteration - 1))

        for ell, noisy_scalar in enumerate(noisy_scalars):
            for _, r_tilde in Lr[ell]:
                for r_hat in (0, 1):
                    r_bar = r_tilde | (r_hat << (iteration - 1))
                    d_bar_ell = (r_bar * params.E + d_bar) & mod_i

                    r_window = r_bar >> start_bit
                    d_window = d_bar_ell >> start_bit
                    scores[d_hat] += evaluate_probability(
                        r_window,
                        d_window,
                        noisy_scalar,
                        start_bit,
                        width,
                        params,
                    )

    best_bit = 0 if scores[0] >= scores[1] else 1
    return d_low | (best_bit << (iteration - 1))


def improved_algorithm_step2(
        iteration: int,
        L: int,
        noisy_scalars: list[int],
        Lr: list[list[tuple[float, int]]],
        d_star: int,
        params: AttackParams,
) -> list[list[tuple[float, int]]]:
    """Paper-style step 2: extend each per-trace candidate list and keep top-L."""
    next_bit = 1 << (iteration - 1)
    mod_i = _mask(iteration)

    updated: list[list[tuple[float, int]]] = []
    for ell, row in enumerate(Lr):
        prev_candidates = [cand for _, cand in row]
        scored: dict[int, float] = {}

        for prev in prev_candidates:
            for add_bit in (0, 1):
                cand = prev | (next_bit if add_bit else 0)
                d_bar_ell = (cand * params.E + d_star) & mod_i
                p = evaluate_probability(cand, d_bar_ell, noisy_scalars[ell], 0, iteration, params)

                scored[cand] = p

        best = sorted(((p, cand) for cand, p in scored.items()), key=lambda x: (-x[0], x[1]))[:L]
        updated.append(best)

    return updated


def benchmark(
        noisy_scalars: list[int],
        d_true: int,
        R: int,
        L: int,
        t: int,
        params: AttackParams) -> int:
    Lr: list[list[tuple[float, int]]] = [[(1.0, 0)] for _ in noisy_scalars]

    d_prefix = 0
    for iteration in range(1, R + 1):
        d_star = improved_algorithm_step1(iteration, t, noisy_scalars, Lr, d_prefix, params)
        if (d_star & _mask(iteration)) != (d_true & _mask(iteration)):
            return iteration - 1

        Lr = improved_algorithm_step2(iteration, L, noisy_scalars, Lr, d_star, params)
        d_prefix = d_star
    return R
