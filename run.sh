#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
    UV="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    echo "uv is not installed yet. Run ./install.sh first." >&2
    exit 1
fi
exec "$UV" run --frozen python cli.py "$@"
