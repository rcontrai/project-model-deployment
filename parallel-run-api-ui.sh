#!/bin/bash
# Script pour lancer l'API et l'UI en parallèle, en prenant soin à ce que tout puisse s'arrêter proprement
# Basé sur https://oneuptime.com/blog/post/2026-01-24-bash-background-processes/view#7-graceful-shutdown


# Track background processes for cleanup
declare -a WORKER_PIDS=()
SHUTDOWN_REQUESTED=false

cleanup() {
    SHUTDOWN_REQUESTED=true
    echo "Shutting down..."
    # Send TERM to all workers
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping worker PID: $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    # Wait for graceful shutdown (with timeout)
    local timeout=10
    local count=0
    while [[ $count -lt $timeout ]]; do
        local running=0
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            fi
        done
        if [[ $running -eq 0 ]]; then
            break
        fi
        sleep 1
        ((count++))
    done
    # Force kill remaining
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing PID: $pid"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    wait
    echo "Shutdown complete"
}

trap cleanup SIGINT SIGTERM

# Lancement des scripts
uvicorn api:app_predict --host 0.0.0.0 --port 8000 &
WORKER_PIDS+=($!)
sleep 3 # Attend que l'API ait bien démarré avant de lancer l'UI
streamlit run ui.py &
WORKER_PIDS+=($!)


echo "Started ${#WORKER_PIDS[@]} workers"
echo "Press Ctrl+C to stop"

# Main loop
while ! $SHUTDOWN_REQUESTED; do
    sleep 1
done