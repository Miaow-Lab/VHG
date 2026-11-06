"""Exact checkpoint budgets for later-round generation; no model imports."""
import math


def portfolio_order(seed_count: int, checkpoint_count: int) -> list[list[tuple[int, int]]]:
    """Interleaved request order as (request index, seed position) per checkpoint."""
    remainder = seed_count % checkpoint_count
    groups = math.gcd(remainder, checkpoint_count)
    offsets = [(group + step * remainder) % checkpoint_count
               for group in range(groups) for step in range(checkpoint_count // groups)]
    order = [[] for _ in range(checkpoint_count)]
    for request in range(seed_count * checkpoint_count):
        cycle, position = divmod(request, seed_count)
        order[(position + offsets[cycle]) % checkpoint_count].append((request, position))
    return order


def allocate_budget(entries: list, checkpoint_count: int, budget: int) -> list[list[tuple[int, list]]]:
    """Equal checkpoint totals, per-seed counts differing by at most one.

    Rotate the unavoidable extra draws across checkpoints. Every returned pair
    is (samples per seed, entries), so the existing batch sampler can be reused.
    """
    if not entries or checkpoint_count < 1 or budget < 1:
        raise ValueError("Need seeds, checkpoints and a positive generation budget.")
    per_checkpoint, remainder = divmod(budget, checkpoint_count)
    if remainder:
        raise ValueError("Generation budget must divide equally across checkpoints.")
    count, extra = divmod(per_checkpoint, len(entries))
    result = []
    for checkpoint, order in enumerate(portfolio_order(len(entries), checkpoint_count)):
        extras = {(checkpoint * extra + i) % len(entries) for i in range(extra)}
        result.append([(n, [entries[i] for _, i in order if count + int(i in extras) == n])
                       for n in (count, count + 1) if n > 0])
    return [[(n, group) for n, group in jobs if group] for jobs in result]
