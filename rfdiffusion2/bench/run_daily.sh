#!/usr/bin/env bash
# "Daily use" configurations: the production contig SHAPE (multiple catalytic
# residues, ~180 designed residues) and the production T=100, run at sizes this
# CPU box can finish. Full-scale L=230 AT T=100 is ~12-24 h per side here and is
# deliberately not attempted -- RFdiffusion2 is normally run on GPU.
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
while pgrep -f "run_all.sh|run_rest.sh" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] starting daily-use batch" >&2
tail -n +2 bench/cases_daily.tsv | while IFS=$'\t' read -r c p l g len T x; do
  [ -z "$c" ] && continue
  grep -q "^$c	" bench/results.tsv && { echo "[skip] $c" >&2; continue; }
  echo "[$(date +%H:%M:%S)] $c  $p  $g  T=$T" >&2
  bash bench/run_case.sh "$c" "$p" "$l" "$g" "$len" "$T" "$x" >> bench/results.tsv
done
echo "[$(date +%H:%M:%S)] DAILY BATCH DONE" >&2
