#!/bin/bash

# Abort script if any error occurs
trap "exit" INT TERM ERR
trap "kill 0" EXIT

HERE="$(pwd)"
INSTALLED="$(which deepcave)"

if ! [[ $INSTALLED ]]; then
    echo "Error: deepcave is not installed." >&2
    exit 1
fi

ROOT_PATH=`
python << END
import deepcave
from pathlib import Path
print(str(Path(deepcave.__file__).parent))
END
`

OPEN=$1
N_WORKERS=$2

# Save config value
CONFIG=$3
if ! [[ $CONFIG ]]; then
    CONFIG="default"
fi
echo "Using config '$CONFIG'"

check_redis() {
    deepcave --check_redis --config "$CONFIG"
}

echo "Checking the configured Redis connection..."

# Use an already running Redis instance, including one published by Docker.
if check_redis > /dev/null 2>&1; then
    echo "Redis server already running. Skipping..."
else
    # Preserve the previous behavior as a fallback for local installations.
    if ! command -v redis-server > /dev/null 2>&1; then
        check_redis || true
        echo "Start Redis with 'docker compose up -d redis' or install redis-server locally." >&2
        exit 1
    fi

    REDIS_ADDRESS=$(deepcave --get_config_value REDIS_ADDRESS --config "$CONFIG")
    REDIS_PORT=$(deepcave --get_config_value REDIS_PORT --config "$CONFIG")

    if [[ "$REDIS_ADDRESS" != "redis://localhost" \
        && "$REDIS_ADDRESS" != "redis://127.0.0.1" \
        && "$REDIS_ADDRESS" != "redis://[::1]" ]]; then
        check_redis || true
        echo "Error: The configured non-local Redis server must be started separately." >&2
        exit 1
    fi

    echo "Redis server is not reachable. Starting a local server..."
    cd "$ROOT_PATH"

    # We remove dump.rdb to avoid issues from previous runs
    file="dump.rdb"
    if [[ -f "$file" ]] ; then
        rm "$file"
    fi

    # We don't want to show the redis-server output
    redis-server --port "$REDIS_PORT" > /dev/null 2>&1 &
    cd "$HERE"

    # Give Redis up to five seconds to accept connections.
    for _ in {1..50}; do
        if check_redis > /dev/null 2>&1; then
            echo "Redis server successfully started."
            break
        fi
        sleep 0.1
    done

    if ! check_redis > /dev/null 2>&1; then
        check_redis || true
        echo "Error: The local Redis server did not become ready." >&2
        exit 1
    fi
fi

# Start n workers in background
for (( i=1; i<=$N_WORKERS; i++ ))
do
	python "$ROOT_PATH/worker.py" --config "$CONFIG" &
done

ADDRESS=$(deepcave --get_config_value DASH_ADDRESS --config "$CONFIG")
PORT=$(deepcave --get_config_value DASH_PORT --config "$CONFIG")
# Open the webserver delayed
if [ "$OPEN" == "true" ]; then
    python "$ROOT_PATH/open.py" --address "$ADDRESS"  --port "$PORT" &
fi

# Start the webserver
python "$ROOT_PATH/server.py" --config "$CONFIG"
