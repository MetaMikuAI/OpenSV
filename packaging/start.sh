#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Install it from https://docs.astral.sh/uv/ and retry.' >&2
    exit 1
fi
if [ ! -x "$ROOT/opensv" ]; then
    echo 'Bundled opensv engine is missing or not executable. Extract the complete archive.' >&2
    exit 1
fi
if [ "${1:-}" = '--cuda' ]; then
    shift
    exec uv run --project "$ROOT/training" --locked --no-group cpu --extra cuda opensv-gui --engine "$ROOT/opensv" "$@"
fi
exec uv run --project "$ROOT/training" --locked opensv-gui --engine "$ROOT/opensv" "$@"
