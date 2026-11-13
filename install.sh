#!/usr/bin/env sh
set -eu

UV_VERSION="0.13.8"
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$1"; }

if command -v uv >/dev/null 2>&1; then
    UV="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    say "Installing uv $UV_VERSION into $HOME/.local/bin (no admin rights needed)"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
    else
        echo "Neither curl nor wget is available. Install one of them, then run ./install.sh again." >&2
        exit 1
    fi
    UV="$HOME/.local/bin/uv"
fi

say "Installing Python 3.11 and the exact package versions (first time: a few minutes)"
"$UV" sync --frozen

if [ "${1:-}" = "--no-data" ]; then
    say "Skipping the data download (run ./run.sh data later)"
else
    say "Downloading and building the data (first time: about 3 minutes)"
    "$UV" run --frozen python -m data_pipeline
fi

say "Checking the installation"
"$UV" run --frozen python cli.py check

say "Done. Next: ./run.sh smoke   (or ./run.sh help)"
