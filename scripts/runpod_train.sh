#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DISHSPACE_ROOT="$ROOT_DIR"
export RUNPOD_VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/runpod-volume/dishspace}"
export DATA_DIR="${DATA_DIR:-$RUNPOD_VOLUME_ROOT/data}"
# Default to OSMesa for headless rendering because it is more reliable than EGL
# across RunPod base images. Users can still override this per-shell.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MODELS_DIR="${MODELS_DIR:-$RUNPOD_VOLUME_ROOT/models}"
export CACHE_DIR="${CACHE_DIR:-$RUNPOD_VOLUME_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "$DATA_DIR/training/runpod" "$MODELS_DIR/dora" "$CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# Fail fast if HF_TOKEN is missing — pi0-base is a gated model
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. The pi0-base model is gated and requires authentication."
  echo "  1. Get a token:  https://huggingface.co/settings/tokens"
  echo "  2. Accept terms: https://huggingface.co/physical-intelligence/pi0-base"
  echo "  3. Run:          export HF_TOKEN=hf_..."
  exit 1
fi

python "$ROOT_DIR/scripts/check_deps.py" --profile train

exec python "$ROOT_DIR/scripts/train.py" \
  --data-dir "$DATA_DIR/training/runpod" \
  --output-dir "$MODELS_DIR/dora/kitchen_v1" \
  "$@"