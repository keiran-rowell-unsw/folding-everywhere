#!/bin/bash
# Dump all fixtures needed by the full-pipeline e2e test for one protein.
# Usage: dump_all.sh <protein_name>
set -e
P="${1:-crambin46}"
cd "$(dirname "$0")"
source /home/lingxusb_gmail_com/esmfold2_venv/bin/activate
for d in dump_modules dump_parcae dump_msa dump_diffusion dump_sampler dump_confidence; do
  echo "=== $d $P ==="
  python $d.py "$P" 2>&1 | grep -vE "ESMC:|🚨|No checkpoint|warnings.warn|UserWarning|Loading checkpoint|\[load\]" | tail -1
done
echo "=== all dumps done for $P ==="
