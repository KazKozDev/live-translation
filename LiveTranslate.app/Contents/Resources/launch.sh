#!/bin/bash
# Launcher for the live-translate overlay. Lives inside the .app bundle at
# <project>/LiveTranslate.app/Contents/Resources/launch.sh and resolves the project
# root in one of two modes:
#
#   1. Portable mode  — the .app sits INSIDE the project folder. The project root is
#      three directories up; derived relatively, so the whole folder is portable: copy
#      it anywhere, on any mac, double-click the app inside it.
#
#   2. Installed mode  — the .app was dragged to /Applications (or anywhere) on its own,
#      detached from the project. The relative path no longer points at the project, so
#      we read the project location from a fixed config file written by install-app.sh:
#          ~/Library/Application Support/LiveTranslate/project_dir
#
# (Run ./setup.sh once inside the project to build the venv.)

CONFIG="$HOME/Library/Application Support/LiveTranslate/project_dir"
RELATIVE="$(cd "$(dirname "$0")/../../.." && pwd)"

if [ -f "$RELATIVE/live_translate_overlay.py" ]; then
    PROJECT_DIR="$RELATIVE"                       # mode 1: app is inside the project
elif [ -f "$CONFIG" ]; then
    PROJECT_DIR="$(cat "$CONFIG")"                # mode 2: fixed location from config
else
    /usr/bin/osascript -e 'display alert "LiveTranslate" message "Не найден проект.\n\nЕсли .app лежит вне папки проекта, запусти один раз:\n  ./install-app.sh\nиз папки проекта — он пропишет путь."'
    exit 1
fi

PY="$PROJECT_DIR/.venv/bin/python"
SCRIPT="$PROJECT_DIR/live_translate_overlay.py"
LOG="$PROJECT_DIR/live_translate_overlay.boot.log"

if [ ! -f "$SCRIPT" ]; then
    /usr/bin/osascript -e 'display alert "LiveTranslate" message "Проект указан, но скрипт не найден:\n'"$SCRIPT"'\n\nПроверь путь в:\n~/Library/Application Support/LiveTranslate/project_dir"'
    exit 1
fi
if [ ! -d "$PROJECT_DIR/live_translation" ]; then
    /usr/bin/osascript -e 'display alert "LiveTranslate" message "Проект указан, но пакет live_translation не найден:\n'"$PROJECT_DIR/live_translation"'\n\nОбнови проект целиком или повтори ./install-app.sh из актуальной папки проекта."'
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

# Apps launched via LaunchServices (double-click) get a minimal PATH that does NOT
# include Homebrew, so whisper can't find `ffmpeg` even when it's installed. Prepend
# the Homebrew bin dirs (arm64 + Intel) so the bundled app behaves like the terminal.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ ! -x "$PY" ]; then
    /usr/bin/osascript -e 'display alert "LiveTranslate" message "Не найден venv:\n'"$PY"'\n\nСоздай окружение и установи зависимости (./setup.sh)."'
    exit 1
fi

{
    echo "===== $(date) :: launch ====="
    exec "$PY" "$SCRIPT" --whisper medium --ollama-model gemma4:12b-mlx "$@"
} >> "$LOG" 2>&1
