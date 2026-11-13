#!/usr/bin/env bash
# Wipe everything the agent has learned. Use after a broken observation/action
# contract - old transitions teach the model the wrong world.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[reset] stopping brain + trainer"
tmux kill-session -t soma_brain 2>/dev/null && echo "  soma_brain stopped"
tmux kill-session -t soma_trainer 2>/dev/null && echo "  soma_trainer stopped"
sleep 2

echo "[reset] removing model, replay buffer, metrics, logs"
rm -fv "$ROOT"/data/replay.f16 "$ROOT"/data/replay.f16.hdr "$ROOT"/data/replay.f16.prio \
       "$ROOT"/data/model.npz "$ROOT"/data/model.npz.tmp* \
       "$ROOT"/data/metrics.jsonl "$ROOT"/data/state.json \
       "$ROOT"/data/brain.log "$ROOT"/data/trainer.log
echo "[reset] done. data/ now:"
ls -la "$ROOT/data" 2>/dev/null | tail -n +2