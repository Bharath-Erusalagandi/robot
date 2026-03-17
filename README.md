# Abluo

Abluo is a kitchen robotics toolkit for grasp planning in sink-like scenes with clutter, wet surfaces, transparent glass, and reflective metal.

It includes:
- synthetic data generation
- local and Modal-oriented fine-tuning workflows
- DishBench-style evaluation
- a FastAPI serving layer with API-key auth
- a Streamlit demo for quick visual validation

## What Is In This Repository

- API server: `src/api/server.py`
- CLI entry point: `src/cli.py`
- data pipeline: `src/data/`
- model + inference: `src/inference/`
- evaluation: `src/evaluation/`
- training/eval scripts: `scripts/`
- demo UI: `demo/app.py`

## Quick Start (Local)

### 1) Create environment and install

```bash
python3 -m venv robot-env
source robot-env/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e ".[dev,demo,gpu]"
```

### 2) Configure environment

```bash
cp .env.example .env
```

Minimum recommended values for local development:
- `DISHSPACE_API_KEY=dev-key-change-me`
- `DEVICE=cpu` (or `cuda`)
- `DEPTH_COMPLETION=none`
- `SEGMENTATION=none`

For real training on pi0-base, set `HF_TOKEN` and ensure your token has access:
- https://huggingface.co/physical-intelligence/pi0-base
- https://huggingface.co/settings/tokens

### 3) Run dependency preflight

```bash
python scripts/check_deps.py --profile all
```

### 4) Start the API

```bash
dishspace serve --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/health
```

## Common Workflows

### Generate synthetic annotations
```bash
dishspace generate --count 5000 --seed 42 --output data/synthetic_annotations.json
```

### Train adapter locally
```bash
dishspace train --quick --output-dir models/dora/kitchen_v1
```

### Evaluate model performance
```bash
dishspace evaluate --local --compare-baseline --adapter models/dora/kitchen_v1/adapter --output data/eval/compare.json
```

### Run grasp planning from an image
```bash
dishspace plan data/example_rgb.png --profile default --robot UR5_realsense --output data/grasp_result.json
```

### Run demo UI
```bash
streamlit run demo/app.py
```
The demo expects the API at http://localhost:8000.

## CLI Commands

The installed CLI command is `dishspace`.

Run help:
```bash
dishspace --help
dishspace train --help
```

## API Overview

Primary endpoints in `src/api/server.py`:
- `POST /grasp_plan`
- `POST /grasp_plan/batch`
- `POST /grasp_plan/ros_trajectory`
- `POST /fine_tune`
- `GET /fine_tune/{job_id}/status`
- `GET /profiles`
- `GET /profiles/{profile_name}`
- `POST /evaluate`
- `GET /usage`
- `GET /health`

Interactive docs:
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Docker

Build and run API:
```bash
docker compose up --build
```

## Runpod

For GPU pods and persistent volume workflows, see `docs/runpod.md`.

## Notes
- Keep secrets in `.env` and never commit credentials.
