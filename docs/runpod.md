# Runpod Setup

This repo is ready to run on a GPU pod with a network volume so training data, adapters, and Hugging Face downloads do not depend on local laptop storage.

## Recommended Pod Setup

Use a GPU pod with:

- A modern NVIDIA GPU such as an RTX 5090.
- A persistent Runpod network volume.
- A PyTorch or CUDA base image with Python 3.11.

Recommended mounted paths:

- Repo checkout: `/workspace/robot`
- Network volume root: `/runpod-volume/dishspace`

## One-Time Bootstrap

From the repo root on the pod:

```bash
bash scripts/setup_runpod.sh --system-deps --install
```

This will:

- Install headless GL/EGL system libraries needed by MuJoCo on a headless pod.
- Create volume-backed directories for data, models, cache, and logs.
- Export environment variables for DishSpace paths.
- Optionally install the project in editable mode with GPU, demo, and dev dependencies.

If the MuJoCo preflight still fails after bootstrap, rerun the system package step explicitly:

```bash
apt-get update && apt-get install -y --no-install-recommends \
	libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri \
	libgles2 libglfw3 libglfw3-dev libglew2.2 libgbm1 libdrm2 libglvnd0 libglx-mesa0 \
	libosmesa6 libosmesa6-dev libglib2.0-0 libglvnd-dev mesa-utils \
	libxrender1 libxext6 libsm6
```

## Environment Layout

After setup, the important paths are:

- `DATA_DIR=/runpod-volume/dishspace/data`
- `MODELS_DIR=/runpod-volume/dishspace/models`
- `CACHE_DIR=/runpod-volume/dishspace/cache`
- `HF_HOME=/runpod-volume/dishspace/cache/huggingface`

## Hugging Face Access

The default base model `physical-intelligence/pi0-base` is gated.

Before any real training run:

1. Accept the model terms at `https://huggingface.co/physical-intelligence/pi0-base`
2. Create a read token at `https://huggingface.co/settings/tokens`
3. Export it on the pod:

```bash
export HF_TOKEN=hf_...
```

## Training

Preflight the environment before training:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python scripts/check_deps.py --profile train
```

The preflight now checks both package imports and gated-model access for `pi0-base`.

Balanced metadata-only dry run:

```bash
bash scripts/runpod_train.sh --dry-run --samples 5000 --no-render --include-failures
```

Use `--no-render` only for dry runs. Full training requires rendered RGB inputs.

Rendered dataset and training run:

```bash
bash scripts/runpod_train.sh --samples 5000 --epochs 5 --include-failures
```

The RunPod wrapper defaults to `MUJOCO_GL=osmesa` because that backend is more
portable across headless pod images than `egl`.

The wrapper stores outputs on the network volume by default:

- Training data under `$DATA_DIR/training/runpod`
- Adapters under `$MODELS_DIR/dora/kitchen_v1`

## API Serving

To launch the API on the pod:

```bash
bash scripts/runpod_serve.sh
```

That wrapper exports the same volume-backed paths and starts the FastAPI app on port 8000.

## Notes

- If you want to avoid large model downloads during local-style heuristic testing, set `DEPTH_COMPLETION=ip_basic` and `SEGMENTATION=none`.
- Keep `.env` secrets on the pod, not in the repo.
- If you switch pods, attach the same Runpod network volume so data and adapters persist.