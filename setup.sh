#!/bin/bash
# One-shot setup for Live Translate: Python deps + system deps + models.
# Usage: ./setup.sh
set -e
cd "$(dirname "$0")"

echo "==> 1/4  Python venv + pip dependencies"
if [ ! -d .venv ]; then
    python3 -m venv .venv
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
