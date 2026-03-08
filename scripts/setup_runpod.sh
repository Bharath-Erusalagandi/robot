#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/runpod-volume/dishspace}"
INSTALL_DEPS="false"
INSTALL_SYSTEM_DEPS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL_DEPS="true"
      shift
      ;;
    --system-deps)
      INSTALL_SYSTEM_DEPS="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export DISHSPACE_ROOT="$ROOT_DIR"
export DATA_DIR="${DATA_DIR:-$VOLUME_ROOT/data}"
export MODELS_DIR="${MODELS_DIR:-$VOLUME_ROOT/models}"
export CACHE_DIR="${CACHE_DIR:-$VOLUME_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$VOLUME_ROOT/logs"

if [[ ! -f "$ROOT_DIR/.env" && -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

if [[ "$INSTALL_SYSTEM_DEPS" == "true" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends \
      libegl1 \
      libegl-mesa0 \
      libgl1 \
      libgl1-mesa-dri \
      libgles2 \
      libglfw3 \
      libglfw3-dev \
      libglew2.2 \
      libgbm1 \
      libdrm2 \
      libglvnd0 \
      libglx-mesa0 \
      libosmesa6 \
      libosmesa6-dev \
      libglib2.0-0 \
      libglvnd-dev \
      mesa-utils \
      libxrender1 \
      libxext6 \
      libsm6
  else
    echo "apt-get not found; install headless GL packages manually for MuJoCo/EGL support." >&2
  fi
fi

if [[ "$INSTALL_DEPS" == "true" ]]; then
  python -m pip install --upgrade pip
  if ! python -m pip install --ignore-installed -e '.[dev,gpu,demo]'; then
    python -m pip install --break-system-packages --ignore-installed -e '.[dev,gpu,demo]'
  fi
fi

cat <<EOF
Runpod environment prepared.

ROOT_DIR=$ROOT_DIR
DATA_DIR=$DATA_DIR
MODELS_DIR=$MODELS_DIR
CACHE_DIR=$CACHE_DIR
HF_HOME=$HF_HOME

Next steps:
  1. Edit .env and add secrets such as SUPABASE_* and HF_TOKEN if needed.
  2. Run: python scripts/check_deps.py --profile train
  3. Run: bash scripts/runpod_train.sh --dry-run --samples 5000 --no-render --include-failures
  4. Run: bash scripts/runpod_train.sh --samples 5000 --epochs 5 --include-failures
EOF