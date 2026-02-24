from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import numpy as np

from engine import Params, simulate
from ga import optimize_rmns

app = FastAPI(title="Bio State Space API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Scenario = Literal["stress", "no-stress"]


# -------------------------
# SIMULATE
# -------------------------
class SimRequest(BaseModel):
    scenario: Scenario = "stress"
    t_max: float = Field(80.0, ge=5.0, le=300.0)
    dt: float = Field(0.02, ge=0.001, le=0.1)
    seed: int = 7

    spike_time: float = 35.0
    spike_amp: float = 8.0
    spike_width: float = 3.0

    # model params
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    alpha_u: float = 3.0
    gamma_b: float = 2.0

    stress_noise_std: float = 2.0
    nostress_noise_std: float = 0.02

    max_points: int = 4000

    # OPTIONAL: GA-optimized weights (if provided)
    rmns_weights: Optional[List[float]] = None  # [wR,wM,wN,wS]


class SimResponse(BaseModel):
    t: List[float]
    x: List[float]
    y: List[float]
    z: List[float]
    spike_index: int
    rmns_weights: Optional[List[float]] = None


@app.post("/simulate", response_model=SimResponse)
def simulate_endpoint(req: SimRequest):
    params = Params(
        sigma=req.sigma,
        rho=req.rho,
        beta=req.beta,
        alpha_u=req.alpha_u,
        gamma_b=req.gamma_b,
        stress_noise_std=req.stress_noise_std,
        nostress_noise_std=req.nostress_noise_std,
    )

    rmns_w = None
    if req.rmns_weights is not None:
        if len(req.rmns_weights) != 4:
            raise ValueError("rmns_weights must be length 4: [wR,wM,wN,wS]")
        rmns_w = np.array(req.rmns_weights, dtype=float)

    t, traj = simulate(
        params=params,
        scenario=req.scenario,
        t_max=req.t_max,
        dt=req.dt,
        seed=req.seed,
        spike_time=req.spike_time,
        spike_amp=req.spike_amp,
        spike_width=req.spike_width,
        rmns_weights=rmns_w,
    )

    # Downsample for browser
    n = len(t)
    if n > req.max_points:
        step = int(np.ceil(n / req.max_points))
        t = t[::step]
        traj = traj[::step]

    spike_index = int(np.searchsorted(t, req.spike_time))
    spike_index = max(0, min(spike_index, len(t) - 1))

    print("SIMULATE scenario:", req.scenario, "| weights:", req.rmns_weights)

    return SimResponse(
        t=t.tolist(),
        x=traj[:, 0].tolist(),
        y=traj[:, 1].tolist(),
        z=traj[:, 2].tolist(),
        spike_index=spike_index,
        rmns_weights=req.rmns_weights,
    )


# -------------------------
# OPTIMIZE
# -------------------------
class OptimizeRequest(BaseModel):
    scenario: Scenario = "no-stress"
    t_max: float = Field(70.0, ge=5.0, le=300.0)
    dt: float = Field(0.01, ge=0.001, le=0.1)
    seed: int = 7

    spike_time: float = 30.0
    spike_amp: float = 12.0
    spike_width: float = 3.0

    # model params
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    alpha_u: float = 3.0
    gamma_b: float = 2.0

    stress_noise_std: float = 2.0
    nostress_noise_std: float = 0.02

    generations: int = Field(12, ge=2, le=60)
    pop_size: int = Field(14, ge=4, le=60)


class OptimizeResponse(BaseModel):
    rmns_weights: List[float]
    best_spread: float


@app.post("/optimize", response_model=OptimizeResponse)
def optimize_endpoint(req: OptimizeRequest):
    params = Params(
        sigma=req.sigma,
        rho=req.rho,
        beta=req.beta,
        alpha_u=req.alpha_u,
        gamma_b=req.gamma_b,
        stress_noise_std=req.stress_noise_std,
        nostress_noise_std=req.nostress_noise_std,
    )

    best_w, best_spread = optimize_rmns(
        params=params,
        scenario=req.scenario,
        t_max=req.t_max,
        dt=req.dt,
        seed=req.seed,
        spike_time=req.spike_time,
        spike_amp=req.spike_amp,
        spike_width=req.spike_width,
        generations=req.generations,
        pop_size=req.pop_size,
    )

    print("OPTIMIZE scenario:", req.scenario, "| best_w:", best_w, "| best_spread:", best_spread)

    return OptimizeResponse(rmns_weights=best_w, best_spread=best_spread)
