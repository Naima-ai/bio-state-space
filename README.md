To run the project:
docker compose up --build

Then:
Open port 5173 for the frontend UI.
(Optional) Open port 8000/docs for the backend API.
For a clear visual difference between Stress and No-stress, I recommend using these demo settings in the UI:
C) DEMO SETTINGS
Recommended parameters:

t_max = 70
dt = 0.01
spike_time = 30
spike_amp = 12
seed = 7

Then:
Click Stress → plot updates
Click No-stress → plot updates

With some other parameter combinations, the visual difference may not be very obvious, but it still works mathematically. To clearly understand the difference, the settings above are recommended.

The Spread proxy value in the UI helps quantify the difference:
-In Stress mode, it generally gives a higher value (more dispersed attractor).
-In No-stress mode, it gives a lower value (tighter and more stable attractor).
