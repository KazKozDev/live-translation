#!/bin/bash
# One-shot setup for Live Translate: Python deps + system deps + models.
# Usage: ./setup.sh
set -e
cd "$(dirname "$0")"

MIN_MACOS_MAJOR=14
MIN_PY_MINOR=12   # Python 3.12+

# Preflight: the pinned MLX / PyTorch wheels only exist for Apple silicon on
# macOS 14+ and Python 3.12+. Fail here with a clear message instead of letting
# pip die later with "No matching distribution found for mlx".
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Error: Live Translation needs a Mac with Apple silicon (found $(uname -s) $(uname -m))." >&2
    exit 1
fi
MACOS_VERSION="$(sw_vers -productVersion)"
if [ "${MACOS_VERSION%%.*}" -lt "$MIN_MACOS_MAJOR" ]; then
    echo "Error: macOS $MIN_MACOS_MAJOR (Sonoma) or newer is required — found macOS $MACOS_VERSION." >&2
    echo "       MLX and PyTorch do not publish wheels for older macOS versions." >&2
    exit 1
fi

py_ok() {
    "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, $MIN_PY_MINOR) else 1)" >/dev/null 2>&1
}

echo "==> 1/4  Python venv + pip dependencies"
if [ -d .venv ]; then
    if ! py_ok ./.venv/bin/python; then
        echo "Error: existing .venv uses $(./.venv/bin/python --version 2>&1); Python 3.$MIN_PY_MINOR+ is required." >&2
        echo "       Remove it (rm -rf .venv) and run ./setup.sh again." >&2
        exit 1
    fi
else
    PYTHON=""
    for candidate in python3.14 python3.13 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: Python 3.$MIN_PY_MINOR+ not found (python3 is $(python3 --version 2>&1))." >&2
        echo "       Install it with: brew install python@3.12" >&2
        exit 1
    fi
    echo "    using $PYTHON ($("$PYTHON" --version 2>&1))"
    "$PYTHON" -m venv .venv
fi
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "==> 2/4  System dependencies (Homebrew: BlackHole audio + Ollama)"
if command -v brew >/dev/null 2>&1; then
    brew bundle --file=Brewfile
else
    echo "    Homebrew not found — install manually: https://brew.sh"
    echo "    then: brew bundle --file=Brewfile"
fi

echo "==> 3/4  Ollama Gemma 4 translation models"
if command -v ollama >/dev/null 2>&1; then
    for model in gemma4:26b-mlx gemma4:e4b-mlx gemma4:12b-mlx; do
        ollama pull "$model" || echo "    skipping $model (run 'ollama serve' and retry if needed)"
    done
else
    echo "    ollama not found — skipping Gemma models"
fi

echo "==> 4/4  Pre-fetch speech models (Whisper medium + turbo MLX)"
./.venv/bin/python - <<'PY' || echo "    models will be downloaded on first run"
from huggingface_hub import snapshot_download
for repo in (
    "mlx-community/whisper-medium-mlx",
    "mlx-community/whisper-large-v3-turbo",
):
    print("    fetching", repo)
    snapshot_download(repo)
PY

echo ""
echo "Done. Run: ./live_translate_overlay.py   (or double-click LiveTranslate.app)"
echo "Remember to route system audio to BlackHole 2ch in System Settings."
