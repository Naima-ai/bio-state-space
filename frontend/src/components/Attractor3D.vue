<template>
  <div class="wrap">
    <div class="panel">
      <h2>RMNS Attractor Demo</h2>

      <div class="row">
        <label>Mode</label>
        <button :class="{active: scenario==='stress'}" @click="setScenario('stress')">Stress</button>
        <button :class="{active: scenario==='no-stress'}" @click="setScenario('no-stress')">No-stress</button>
      </div>

      <div class="row">
        <label>t_max</label>
        <input type="number" v-model.number="tMax" min="10" max="300" step="5" />
      </div>

      <div class="row">
        <label>dt</label>
        <input type="number" v-model.number="dt" min="0.001" max="0.1" step="0.005" />
      </div>

      <div class="row">
        <label>Spike time</label>
        <input type="number" v-model.number="spikeTime" min="0" :max="tMax" step="1" />
      </div>

      <div class="row">
        <label>Spike amp</label>
        <input type="number" v-model.number="spikeAmp" min="0" max="30" step="0.5" />
      </div>

      <div class="row">
        <label>Seed</label>
        <input type="number" v-model.number="seed" step="1" />
      </div>

      <div class="row">
        <button class="run" @click="run" :disabled="loading">
          {{ loading ? "Running…" : "Run / Update" }}
        </button>
      </div>

      <pre class="metrics">{{ metrics }}</pre>
    </div>

    <div class="plot">
      <div ref="plotEl" class="plotEl"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import Plotly from "plotly.js-dist-min";

const plotEl = ref(null);

const scenario = ref("stress");
const tMax = ref(80);
const dt = ref(0.02);
const spikeTime = ref(35);
const spikeAmp = ref(8);
const seed = ref(7);
const loading = ref(false);
const metrics = ref("");

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * Make mode switching visibly update the plot:
 * - sets scenario
 * - triggers run() immediately
 */
function setScenario(val) {
  scenario.value = val;
  run(); // auto-update on toggle
}

/**
 * Plotting change:
 * - Use marker "cloud" rendering for trajectories (much easier to see spread differences)
 * - Keep spike marker + label
 * - Keep spread metric so you can prove stress > no-stress
 */
async function run() {
  if (!plotEl.value) return;

  loading.value = true;
  metrics.value = "";

  try {
    const body = {
      scenario: scenario.value,
      t_max: tMax.value,
      dt: dt.value,
      seed: seed.value,
      spike_time: spikeTime.value,
      spike_amp: spikeAmp.value,
      spike_width: 3.0,
      max_points: 4000,

      // OPTIONAL: force visible difference even if backend defaults change later
      // stress_noise_std: 2.0,
      // nostress_noise_std: 0.02,
    };

    const res = await fetch(`${API_BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API error ${res.status}: ${txt}`);
    }

    const data = await res.json();

    const x = data.x || [];
    const y = data.y || [];
    const z = data.z || [];
    const idxRaw = Number.isFinite(data.spike_index) ? data.spike_index : 0;
    const idx = Math.max(0, Math.min(idxRaw, x.length - 1));

    if (x.length < 10) {
      throw new Error(`Too few points returned (${x.length}). Check backend.`);
    }

    // Split into phases: before spike and after spike
    const x1 = x.slice(0, idx),
      y1 = y.slice(0, idx),
      z1 = z.slice(0, idx);
    const x2 = x.slice(idx),
      y2 = y.slice(idx),
      z2 = z.slice(idx);

    // Basic metrics (spread proxy)
    const meanX = x.reduce((a, b) => a + b, 0) / x.length;
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    const meanZ = z.reduce((a, b) => a + b, 0) / z.length;

    let spread = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - meanX,
        dy = y[i] - meanY,
        dz = z[i] - meanZ;
      spread += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    spread /= x.length;

    const zPeak = Math.max(...z);

    metrics.value =
      `API_BASE: ${API_BASE}\n` +
      `Mode sent: ${scenario.value}\n` +
      `Points: ${x.length}\n` +
      `Spike index: ${idx}\n` +
      `Spread proxy: ${spread.toFixed(3)}\n` +
      `Peak z: ${zPeak.toFixed(3)}\n`;

    // MAIN CHANGE: markers cloud (shows thickness/spread clearly)
    const traces = [
      {
        type: "scatter3d",
        mode: "markers",
        name: "Before spike",
        x: x1,
        y: y1,
        z: z1,
        marker: { size: 2, opacity: 0.25 },
      },
      {
        type: "scatter3d",
        mode: "markers",
        name: "After spike",
        x: x2,
        y: y2,
        z: z2,
        marker: { size: 2, opacity: 0.35 },
      },
      {
        type: "scatter3d",
        mode: "markers+text",
        name: "Spike",
        x: [x[idx]],
        y: [y[idx]],
        z: [z[idx]],
        text: ["Spike"],
        textposition: "top center",
        marker: { size: 5, opacity: 1.0 },
        showlegend: true,
      },
    ];

    const layout = {
      title: "Biological State Space (RMNS Attractor)",
      height: 720,
      margin: { l: 0, r: 0, t: 50, b: 0 },
      scene: {
        xaxis: { title: "x: Autonomic Tone" },
        yaxis: { title: "y: Metabolic Flux" },
        zaxis: { title: "z: Inflammation" },
      },
      legend: { x: 0.01, y: 0.99 },
    };

    await Plotly.react(plotEl.value, traces, layout, { responsive: true });
  } catch (e) {
    metrics.value = `Error: ${e.message}`;
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  run(); // initial render
});
</script>

<style scoped>
.wrap {
  display: grid;
  grid-template-columns: 320px 1fr;
  height: 100vh;
}
.panel {
  padding: 16px;
  border-right: 1px solid #ddd;
  overflow: auto;
}
.row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin: 10px 0;
}
label {
  width: 90px;
  font-size: 14px;
}
button {
  padding: 6px 10px;
  border: 1px solid #999;
  background: #fff;
  cursor: pointer;
}
button.active {
  border-color: #333;
  font-weight: 600;
}
button.run {
  width: 100%;
  padding: 10px;
}
.metrics {
  margin-top: 12px;
  background: #f6f6f6;
  padding: 10px;
  border: 1px solid #e2e2e2;
  font-size: 12px;
  white-space: pre-wrap;
}
.plot {
  padding: 10px;
}
.plotEl {
  width: 100%;
  height: 100%;
}
</style>
