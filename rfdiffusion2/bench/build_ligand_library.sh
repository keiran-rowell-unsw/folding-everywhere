#!/usr/bin/env bash
# Build a per-LIGAND topology library from every mcsa_41 input.
#
# Why a reference run per PDB rather than recomputation: `get_atom_frames` breaks
# priority ties by CPython set-iteration order, and the recomputation path in
# gen_ligand_bonds.py has a different insertion sequence than the pipeline. The
# disagreement is NOT the "1 of 50" the docs record -- measured across ten inputs
# it ranges to 35 of 96 atoms (M0093_1dqa). So each ligand's frames are taken
# from an actual reference run of the file it appears in.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
REF=$ROOT/../ref_RFdiffusion2
IN=$REF/rf_diffusion/benchmark/input/mcsa_41
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE CUDA_VISIBLE_DEVICES=
export PYTHONHASHSEED=1 PYTORCH_JIT=0 RFD2_PINNED=1 PYTHONPATH=$REF RFD2_REF=$REF
mkdir -p bench/ligand_runs

for pdb in "$IN"/*.pdb; do
  prot=$(basename "$pdb" .pdb)
  out=bench/ligand_runs/$prot
  [ -f "fixtures/ligand/$prot.safetensors" ] && \
    .venv/bin/python -c "
from safetensors import safe_open;import sys
f=safe_open('fixtures/ligand/$prot.safetensors','pt')
sys.exit(0 if any(k.endswith('.atom_names') for k in f.keys()) else 1)" 2>/dev/null && \
    { echo "[skip] $prot (already has named sidecar)"; continue; }

  # ligands = every HETATM resname except the ORI marker, in file order
  ligs=$(grep '^HETATM' "$pdb" | cut -c18-20 | tr -d ' ' | grep -v '^ORI$' | awk '!seen[$0]++' | paste -sd, -)
  [ -z "$ligs" ] && { echo "[skip] $prot (no ligands)"; continue; }
  # first CA residue = a valid motif anchor for a minimal contig
  anchor=$(grep '^ATOM' "$pdb" | awk 'substr($0,13,4)==" CA "{c=substr($0,22,1);r=substr($0,23,4)+0;print c r;exit}')
  [ -z "$anchor" ] && { echo "[skip] $prot (no CA)"; continue; }
  ch=${anchor:0:1}; rn=${anchor:1}
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] $prot  ligands=$ligs  anchor=$ch$rn"
  RFD2_DUMP_RFI=$out/rfi.safetensors .venv/bin/python python/run_reference.py --config-name=aa \
    inference.ckpt_path=$REF/rf_diffusion/model_weights/RFD_173.pt \
    inference.input_pdb="$pdb" "inference.ligand='$ligs'" \
    "contigmap.contigs=['5,$ch$rn-$rn,5']" \
    inference.contig_as_guidepost=False inference.num_designs=1 \
    inference.deterministic=True inference.idealize_sidechain_outputs=False \
    inference.write_trb_indep=False diffuser.T=1 \
    inference.output_prefix=$out/design > "$out/out.txt" 2> "$out/log.txt"
  if [ ! -f "$out/rfi.safetensors" ]; then
    echo "    FAILED: $(grep -aoE '[A-Za-z]*(Error|Exception)[^\"]{0,80}' "$out/log.txt" | tail -1)"
    continue
  fi
  RFD2_ATOM_FRAMES=$out/rfi.safetensors .venv/bin/python python/gen_ligand_bonds.py \
    "$pdb" "$ligs" 2>&1 | grep -E "atom_frames:" | sed 's/^/    /'
done
echo "[$(date +%H:%M:%S)] LIGAND LIBRARY RUNS DONE"
