#!/bin/bash
# Install LiveTranslate.app into /Applications, detached from the project folder.
#
# Use this when you want the app to live in /Applications on its own (Launchpad,
# Spotlight, Dock) while the project (venv + models + script) stays where it is.
# The launcher can't derive the project path relatively anymore, so we record it
# in a fixed config file that launch.sh reads in "installed mode".
#
# If instead you just want a portable folder you copy around as a unit, you don't
# need this — keep the .app inside the project and double-click it there.
#
# Usage:  ./install-app.sh            # installs to /Applications
#         ./install-app.sh ~/Apps     # installs to a custom folder
set -e
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
DEST="${1:-/Applications}"
APP="LiveTranslate.app"
CONFIG_DIR="$HOME/Library/Application Support/LiveTranslate"
CONFIG="$CONFIG_DIR/project_dir"

if [ ! -d "$APP" ]; then
    echo "Error: $APP not found next to the script ($PROJECT_DIR)." >&2
    exit 1
fi
if [ ! -f "$PROJECT_DIR/live_translate_overlay.py" ]; then
    echo "Error: run the script from the project folder (live_translate_overlay.py not found)." >&2
    exit 1
fi
if [ ! -d "$PROJECT_DIR/live_translation" ]; then
    echo "Error: run the script from the full project folder (live_translation/ not found)." >&2
    exit 1
fi

echo "==> 1/3  Writing project path to config"
mkdir -p "$CONFIG_DIR"
printf '%s\n' "$PROJECT_DIR" > "$CONFIG"
echo "    $CONFIG -> $PROJECT_DIR"

echo "==> 2/3  Copying $APP to $DEST"
rm -rf "$DEST/$APP"
cp -R "$APP" "$DEST/$APP"

echo "==> 3/3  Re-signing (ad-hoc) the copy in $DEST"
# Resources changed across the copy boundary; re-seal so LaunchServices runs it.
codesign --force --deep -s - "$DEST/$APP"
codesign --verify --deep --strict "$DEST/$APP" && echo "    signature ok"

echo ""
echo "Done. $DEST/$APP now launches the project from:"
echo "    $PROJECT_DIR"
echo ""
echo "If you move the project, re-run ./install-app.sh from the new location"
echo "(or update the path in $CONFIG)."
