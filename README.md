# VLA Dataset Pipeline (Isaac Sim + MCP)

This repository collects VLA-style robot manipulation episodes from Isaac Sim via an MCP socket bridge.

## What It Does

- Connects from WSL2 to Isaac Sim running on Windows through MCP.
- Drives a Franka manipulator with a pick-and-place controller.
- Randomizes cube positions and lighting.
- Records synchronized multi-camera observations:
  - wrist (gripper tip)
  - top
  - front
  - side
- Saves episodes to HDF5 (LeRobot-style structure) with motion validity checks.
- Optionally exports per-camera MP4 videos.

## Project Files

- `orchestrate_vla.py`: WSL2-side orchestrator (socket client, episode loop, manifest generation).
- `vla_collector.py`: Script executed inside Isaac Sim via MCP (`execute_script`).
- `test_connection.py`: Preflight connection and command test.
- `visualize_dataset.py`: Basic visualization utility for saved episodes.

## Runtime Architecture

- Windows:
  - Isaac Sim (GUI / physics / rendering)
  - Isaac Sim MCP server bridge
- WSL2:
  - Python orchestrator + utilities in this repo

## Quick Start

1. Start Isaac Sim on Windows and load your scene (`/World/Franka`, `/World/Obs/*` cubes).
2. Ensure the MCP bridge is running and reachable from WSL2.
3. In WSL2, run a preflight:

```bash
python test_connection.py
```

4. Collect data:

```bash
python orchestrate_vla.py --episodes 10 --max-steps 400
```

## Key Environment Variables

- Connection:
  - `ISAAC_SIM_HOST`
  - `ISAAC_SIM_PORT`
- Collection behavior:
  - `VLA_TASK_MODE`: `pick_place` (default), `stacking`, `random`
  - `VLA_TARGET_FRAMES` (default `120`)
  - `VLA_MIN_VALID_FRAMES` (default `80`)
  - `VLA_REQUIRE_CONTROLLER_DONE` (default `1`)
- Validation thresholds:
  - `VLA_MIN_JOINT_STEP_NORM`
  - `VLA_MIN_EE_TRANSLATION_STEP`
  - `VLA_MIN_JOINT_VEL_ABS_MAX`
  - `VLA_MIN_GRIPPER_DELTA`
  - `VLA_MIN_WRIST_RGB_MEAN_DELTA`
  - `VLA_MIN_ANY_RGB_MEAN_DELTA`
- Video output:
  - `VLA_SAVE_EPISODE_VIDEOS` (default `1`)
  - `VLA_VIDEO_FPS` (default `20`)

## Output

- `episode_XXXXXX.hdf5`
- `dataset_manifest.json`
- `videos/episode_XXXXXX_{wrist,top,front,side}.mp4` (if enabled)

## Notes

- Episodes that fail real-motion validation are rejected by default.
- Set `VLA_SAVE_INVALID_EPISODES=1` only for debugging.
- If MP4 encoding via `imageio` is unavailable in Isaac runtime, OpenCV fallback is used.
