#!/usr/bin/env bash
# SOMA — start / stop / inspect the agent (bot, brain, trainer).
#
# Layout rule from NotionPrompt: everything lives in /home/opc/ai_workspace/soma,
# every tmux session is prefixed with soma_.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MC_HOST="${SOMA_MC_HOST:-127.0.0.1}"
MC_PORT="${SOMA_MC_PORT:-25565}"
IPC_PORT="${SOMA_IPC_PORT:-8787}"

setup_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    echo "[soma] creating venv"
    python3 -m venv "$VENV"
  fi
  if ! "$VENV/bin/python" -c "import numpy" 2>/dev/null; then
    echo "[soma] installing numpy"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$ROOT/agent/requirements.txt"
  fi
}

setup_bot() {
  if [ ! -d "$ROOT/bot/node_modules" ]; then
    echo "[soma] npm install"
    (cd "$ROOT/bot" && npm install --no-audit --no-fund)
  fi
}

start_bot() {
  setup_bot
  if tmux has-session -t soma_bot 2>/dev/null; then echo "[soma] bot already running"; return; fi
  tmux new-session -d -s soma_bot "cd '$ROOT/bot' && SOMA_MC_HOST=$MC_HOST SOMA_MC_PORT=$MC_PORT SOMA_IPC_PORT=$IPC_PORT nice -n 5 node index.js 2>&1 | tee -a '$ROOT/data/bot.log'"
  echo "[soma] bot started (tmux: soma_bot)"
}

start_brain() {
  setup_venv
  if tmux has-session -t soma_brain 2>/dev/null; then echo "[soma] brain already running"; return; fi
  tmux new-session -d -s soma_brain "cd '$ROOT/agent' && SOMA_IPC_PORT=$IPC_PORT nice -n 5 taskset -c 2-3 '$VENV/bin/python' brain.py 2>&1 | tee -a '$ROOT/data/brain.log'"
  echo "[soma] brain started (tmux: soma_brain)"
}

start_trainer() {
  setup_venv
  if tmux has-session -t soma_trainer 2>/dev/null; then echo "[soma] trainer already running"; return; fi
  tmux new-session -d -s soma_trainer "cd '$ROOT/agent' && nice -n 15 taskset -c 0-1 '$VENV/bin/python' trainer.py 2>&1 | tee -a '$ROOT/data/trainer.log'"
  echo "[soma] trainer started (tmux: soma_trainer)"
}

case "${1:-status}" in
  setup)   setup_venv; setup_bot ;;
  bot)     start_bot ;;
  brain)   start_brain ;;
  trainer) start_trainer ;;
  all)     mkdir -p "$ROOT/data"; start_bot; start_brain; start_trainer ;;
  stop)
    for s in soma_bot soma_brain soma_trainer; do
      tmux kill-session -t "$s" 2>/dev/null && echo "[soma] stopped $s"
    done ;;
  status)  "${BASH_SOURCE[0]}" _status ;;
  _status)
    echo "=== SOMA status ==="
    for s in soma_bot soma_brain soma_trainer; do
      if tmux has-session -t "$s" 2>/dev/null; then echo "  $s: RUNNING"; else echo "  $s: stopped"; fi
    done
    echo "--- last metrics ---"
    tail -3 "$ROOT/data/metrics.jsonl" 2>/dev/null || echo "  (none yet)"
    ;;
  *) echo "usage: $0 {setup|bot|brain|trainer|all|stop|status}"; exit 1 ;;
esac