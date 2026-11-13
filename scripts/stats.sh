#!/usr/bin/env bash
# Server + agent telemetry. Read-only.
set -u
MC_DIR="${MC_DIR:-/home/opc/ai_workspace/create-aeronautics-server}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== host ==="
uptime
free -m | head -2
echo
echo "=== minecraft server ==="
pgrep -af "user_jvm_args" | cut -c1-120 || echo "  not running"
grep -h "Time|Running|Can't keep up|Watching" "$MC_DIR/logs/latest.log" 2>/dev/null | tail -5 || true
echo
echo "=== soma processes ==="
ps -eo pid,ni,psr,pcpu,pmem,rss,etime,comm,args --sort=-rss \
  | grep -E "soma|brain.py|trainer.py|node index.js" | grep -v grep \
  | awk '{printf "  pid=%s nice=%s cpu=%s%% mem=%sMB %s\n",$1,$2,$5,$6,$9}'
echo
echo "=== agent metrics ==="
tail -5 "$ROOT/data/metrics.jsonl" 2>/dev/null || echo "  (none yet)"
echo
echo "=== logs (tail) ==="
for f in bot brain trainer; do
  [ -f "$ROOT/data/$f.log" ] && { echo "--- $f ---"; tail -6 "$ROOT/data/$f.log"; }
done