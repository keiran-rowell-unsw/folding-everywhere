#!/usr/bin/env bash
# Wait for the main sweep, then re-run any case that has no good row, plus the
# extra (num_designs) cases. Never edit run_case.sh while this is running:
# bash reads a script by byte offset, and rewriting it mid-execution makes the
# running shell resume at a stale offset (that is what killed p07_L69).
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
while pgrep -f run_all.sh >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] main sweep finished; filling gaps" >&2

# drop rows that are failures, so those cases are retried
grep -avE "	(REF_FAIL|LIGAND_FAIL|PORT_FAIL|LINECOUNT_MISMATCH|MISSING_DESIGN)" \
  bench/results.tsv > bench/.results.keep && mv bench/.results.keep bench/results.tsv

cat bench/cases.tsv > /tmp/all_cases.tsv
tail -n +2 bench/cases_extra.tsv >> /tmp/all_cases.tsv
tail -n +2 /tmp/all_cases.tsv | while IFS=$'\t' read -r c p l g len T x; do
  [ -z "$c" ] && continue
  grep -q "^$c	" bench/results.tsv && continue
  grep -q "^$c#" bench/results.tsv && continue
  echo "[$(date +%H:%M:%S)] retry/extra: $c  $p  $g  T=$T  $x" >&2
  bash bench/run_case.sh "$c" "$p" "$l" "$g" "$len" "$T" "$x" >> bench/results.tsv
done
echo "[$(date +%H:%M:%S)] GAP FILL DONE" >&2
