from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import numpy as np

from engine import Params, simulate

app = FastAPI(title="Bio State Space API", version="1.0")

# Allow Vue dev server + LoopQ host later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Scenario = Literal["stress", "no-stress"]

class SimRequest(BaseModel):
    scenario: Scenario = "stress"
    t_max: float = Field(80.0, ge=5.0, le=300.0)
    dt: float = Field(0.02, ge=0.001, le=0.1)
    seed: int = 7

    spike_time: float = 35.0
    spike_amp: float = 8.0
    spike_width: float = 3.0

    # optional override of model params
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    alpha_u: float = 3.0
    gamma_b: float = 2.0

    stress_noise_std: float = 0.8
    nostress_noise_std: float = 0.05

    # frontend can request fewer points to keep browser fast
    max_points: int = 4000


class SimResponse(BaseModel):
    t: List[float]
    x: List[float]
    y: List[float]
    z: List[float]
    spike_index: int


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

    t, traj = simulate(
        params=params,
        scenario=req.scenario,
        t_max=req.t_max,
        dt=req.dt,
        seed=req.seed,
        spike_time=req.spike_time,
        spike_amp=req.spike_amp,
        spike_width=req.spike_width,
    )

    # Downsample if too many points (important for browser + Codespaces)
    n = len(t)
    if n > req.max_points:
        step = int(np.ceil(n / req.max_points))
        t = t[::step]
        traj = traj[::step]

    spike_index = int(np.searchsorted(t, req.spike_time))
    spike_index = max(0, min(spike_index, len(t) - 1))

    return SimResponse(
        t=t.tolist(),
        x=traj[:, 0].tolist(),
        y=traj[:, 1].tolist(),
        z=traj[:, 2].tolist(),
        spike_index=spike_index,
    )
