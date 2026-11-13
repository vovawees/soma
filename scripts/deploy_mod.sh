#!/usr/bin/env bash
# Rebuild the mod, restart the MC server with it, restart the brain.
# Usage: scripts/deploy_mod.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MC_DIR="${MC_DIR:-/home/opc/ai_workspace/create-aeronautics-server}"
JAR="$ROOT/mod/build/soma-0.1.0.jar"

echo "[deploy] building mod"
bash "$ROOT/mod/build.sh" || exit 1

echo "[deploy] stopping server"
for p in $(pgrep -x java); do
  ps -p "$p" -o args= | grep -q "user_jvm_args" && kill "$p"
done
sleep 6
tmux kill-session -t create_aeronautics_server 2>/dev/null
sleep 1

echo "[deploy] installing jar"
cp "$JAR" "$MC_DIR/mods/"

echo "[deploy] starting server"
cd "$MC_DIR"
mv logs/latest.log logs/latest.prev.log 2>/dev/null
./start.sh >/dev/null
for i in $(seq 1 120); do grep -q "Done (" logs/latest.log 2>/dev/null && break; sleep 5; done
echo "[deploy] booted=$(grep -c 'Done (' logs/latest.log 2>/dev/null)"

echo "[deploy] enabling agent"
tmux send-keys -t create_aeronautics_server "soma start" Enter
sleep 6

echo "[deploy] restarting brain"
tmux kill-session -t soma_brain 2>/dev/null
rm -f "$ROOT/data/brain.log"
tmux new-session -d -s soma_brain "cd $ROOT/agent && SOMA_IPC_PORT=8788 nice -n 5 ../.venv/bin/python -u brain.py 2>&1 | tee -a ../data/brain.log"

echo "[deploy] done, waiting 40s"
sleep 40
grep -i "\[soma\]" "$MC_DIR/logs/latest.log" | tail -4
echo "--- metrics ---"
tail -2 "$ROOT/data/metrics.jsonl" 2>/dev/null || echo "  (none)"