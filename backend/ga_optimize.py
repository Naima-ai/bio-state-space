import random
import numpy as np
from engine import simulate, Params

def calculate_spread(traj):
    mean = traj.mean(axis=0)
    distances = np.linalg.norm(traj - mean, axis=1)
    return float(np.mean(distances))

def evaluate_fitness(rmns_weights):
    params = Params()

    t, traj = simulate(
        params=params,
        scenario="no-stress",
        t_max=80.0,
        dt=0.02,
        seed=7,
        spike_time=35.0,
        spike_amp=8.0,
        spike_width=3.0,
        rmns_weights=np.array(rmns_weights, dtype=float),
    )

    # optionally score only post-spike for "recovery"
    idx = int(np.searchsorted(t, 35.0))
    post = traj[idx:] if idx < len(traj) else traj

    spread = calculate_spread(post)
    return 1.0 / (1.0 + spread)

def mutate(chromosome, mutation_rate=0.1):
    mutated = chromosome.copy()
    for i in range(4):
        if random.random() < mutation_rate:
            mutated[i] += random.uniform(-0.2, 0.2)
            mutated[i] = max(0.0, min(3.5, mutated[i]))  # wider range than 0..1
    return mutated

def crossover(p1, p2):
    return [(a + b) / 2.0 for a, b in zip(p1, p2)]

def run_evolution(generations=20, pop_size=10):
    population = [[random.uniform(0.0, 2.0) for _ in range(4)] for _ in range(pop_size)]

    for gen in range(generations):
        scored = [(evaluate_fitness(ind), ind) for ind in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        best_fitness, best_rmns = scored[0]
        best_spread = (1.0 / best_fitness) - 1.0
        print(f"Gen {gen} | Best Spread: {best_spread:.4f} | weights: {[round(v,3) for v in best_rmns]}")

        next_gen = [ind for score, ind in scored[:max(2, int(pop_size * 0.2))]]
        while len(next_gen) < pop_size:
            p1 = random.choice(scored[:5])[1]
            p2 = random.choice(scored[:5])[1]
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen

    return best_rmns, best_spread
