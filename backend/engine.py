from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.integrate import solve_ivp


@dataclass
class Params:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    alpha_u: float = 3.0   # RMNS forcing strength into y_dot
    gamma_b: float = 2.0   # biomarker forcing strength into z_dot

    # stress vs no-stress noise
    stress_noise_std: float = 0.8
    nostress_noise_std: float = 0.05


def rmns_controls(t: float, scenario: str) -> np.ndarray:
    """
    u(t) = [R, M, N, S]
    These are the PREDEFINED RMNS signals (from your current demo).
    """
    if scenario == "stress":
        R = 0.4 + 0.2 * np.sin(0.3 * t) + 0.2 * np.sin(2.3 * t)
        M = 0.2 + 0.6 * (np.sin(0.15 * t) > 0.8)
        N = 0.5 + 0.3 * np.sin(0.5 * t + 1.0)
        S = 0.9
    else:
        R = 0.8 + 0.1 * np.sin(0.2 * t)
        M = 0.4 + 0.3 * (np.sin(0.12 * t) > 0.6)
        N = 0.7 + 0.1 * np.sin(0.35 * t)
        S = 0.2

    return np.array([R, M, N, S], dtype=float)


def f_rmns(u: np.ndarray, rmns_weights: Optional[np.ndarray] = None) -> float:
    """
    RMNS -> scalar forcing term.
    If rmns_weights is provided, it must be [wR,wM,wN,wS] and replaces the fixed weights.
    """
    R, M, N, S = u

    if rmns_weights is None:
        # your original fixed formula
        return float((0.9 * R + 0.6 * N + 0.4 * M) - (1.2 * S))

    wR, wM, wN, wS = map(float, rmns_weights)
    return float((wR * R + wN * N + wM * M) - (wS * S))


def biomarker_inflammation_spike(t: float, spike_time: float, amp: float, width: float) -> float:
    """Gaussian spike centered at spike_time."""
    return float(amp * np.exp(-0.5 * ((t - spike_time) / width) ** 2))


def g_bio(z: float, b_inflammation: float) -> float:
    """Simple feedback: spike + mild state dependence."""
    return float(b_inflammation + 0.05 * z)


def simulate(
    params: Params,
    scenario: str,
    t_max: float = 80.0,
    dt: float = 0.02,
    x0: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 7,
    spike_time: float = 35.0,
    spike_amp: float = 8.0,
    spike_width: float = 3.0,
    rmns_weights: Optional[np.ndarray] = None,  # <-- NEW (GA can pass weights here)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns time array t (T,) and trajectory traj (T,3).
    If rmns_weights is provided, it influences the forcing term through f_rmns.
    """
    t_eval = np.arange(0.0, t_max, dt, dtype=float)
    n = len(t_eval)

    # Deterministic noise for reproducibility
    rng = np.random.default_rng(seed)
    noise_std = params.stress_noise_std if scenario == "stress" else params.nostress_noise_std
    noise = rng.normal(0.0, noise_std, size=(n, 3)).astype(float)

    def system(t: float, s: np.ndarray) -> np.ndarray:
        x, y, z = s

        u = rmns_controls(t, scenario)
        u_term = params.alpha_u * f_rmns(u, rmns_weights=rmns_weights)

        b = biomarker_inflammation_spike(t, spike_time=spike_time, amp=spike_amp, width=spike_width)
        b_term = params.gamma_b * g_bio(z=z, b_inflammation=b)

        dx = params.sigma * (y - x)
        dy = x * (params.rho - z) - y + u_term
        dz = x * y - params.beta * z + b_term

        # Add deterministic noise by indexing closest t
        idx = int(np.clip(round(t / dt), 0, n - 1))
        dx += noise[idx, 0]
        dy += noise[idx, 1]
        dz += noise[idx, 2]

        return np.array([dx, dy, dz], dtype=float)

    sol = solve_ivp(
        fun=system,
        t_span=(t_eval[0], t_eval[-1]),
        y0=np.array(x0, dtype=float),
        t_eval=t_eval,
        method="RK23",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    traj = sol.y.T
    return sol.t, traj
