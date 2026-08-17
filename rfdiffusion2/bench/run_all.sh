#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
tail -n +2 bench/cases.tsv | while IFS=$'\t' read -r c p l g len T x; do
  [ -z "$c" ] && continue
  grep -q "^$c	" bench/results.tsv && { echo "[skip done] $c" >&2; continue; }
  echo "[$(date +%H:%M:%S)] $c  $p  L-contig=$g  T=$T  $x" >&2
  bash bench/run_case.sh "$c" "$p" "$l" "$g" "$len" "$T" "$x" >> bench/results.tsv
done
echo "[$(date +%H:%M:%S)] ALL CASES DONE" >&2
