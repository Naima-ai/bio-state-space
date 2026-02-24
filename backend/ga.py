import random
import numpy as np
from engine import simulate, Params


def spread_proxy(traj: np.ndarray) -> float:
    """Average distance from mean state."""
    mu = traj.mean(axis=0)
    return float(np.mean(np.linalg.norm(traj - mu, axis=1)))


def fitness(weights4, params: Params, scenario: str, t_max: float, dt: float, seed: int,
            spike_time: float, spike_amp: float, spike_width: float) -> float:
    """
    Higher fitness is better.
    We minimize spread AFTER the spike (recovery stability).
    """
    t, traj = simulate(
        params=params,
        scenario=scenario,
        t_max=t_max,
        dt=dt,
        seed=seed,
        spike_time=spike_time,
        spike_amp=spike_amp,
        spike_width=spike_width,
        rmns_weights=np.array(weights4, dtype=float),
    )

    idx = int(np.searchsorted(t, spike_time))
    post = traj[idx:] if idx < len(traj) else traj

    spread = spread_proxy(post)
    return 1.0 / (1.0 + spread)


def mutate(chromosome, mutation_rate=0.2):
    out = chromosome.copy()
    for i in range(4):
        if random.random() < mutation_rate:
            out[i] += random.uniform(-0.25, 0.25)
            out[i] = max(0.0, min(3.5, out[i]))
    return out


def crossover(p1, p2):
    return [(a + b) / 2.0 for a, b in zip(p1, p2)]


def optimize_rmns(
    params: Params,
    scenario: str,
    t_max: float,
    dt: float,
    seed: int,
    spike_time: float,
    spike_amp: float,
    spike_width: float,
    generations: int = 12,
    pop_size: int = 14,
):
    """
    Returns:
      best_weights = [wR,wM,wN,wS]
      best_spread  = spread value AFTER spike for the best solution
    """
    population = [[random.uniform(0.0, 2.0) for _ in range(4)] for _ in range(pop_size)]

    best_fit = -1e9
    best_w = None

    for _ in range(generations):
        scored = [(fitness(ind, params, scenario, t_max, dt, seed, spike_time, spike_amp, spike_width), ind)
                  for ind in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best_w = scored[0][1]

        elites = [ind for _, ind in scored[:max(2, int(pop_size * 0.25))]]

        nxt = elites[:]
        while len(nxt) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = mutate(crossover(p1, p2))
            nxt.append(child)

        population = nxt

    best_spread = (1.0 / best_fit) - 1.0
    return [float(v) for v in best_w], float(best_spread)
