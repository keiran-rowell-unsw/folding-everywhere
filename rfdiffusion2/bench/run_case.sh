#!/usr/bin/env bash
# One benchmark case: pinned reference -> ligand sidecar from THAT run -> port -> diff.
# Usage: run_case.sh <case> <protein> <ligands> <contig> <length|-> <T> <extra|->
set -uo pipefail
CASE=$1; PROT=$2; LIGS=$3; CONTIG=$4; LEN=$5; T=$6; EXTRA=$7
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
REF=$ROOT/../ref_RFdiffusion2
PDB=$REF/rf_diffusion/benchmark/input/mcsa_41/$PROT.pdb
OUT=$ROOT/bench/runs/$CASE; mkdir -p "$OUT/ref" "$OUT/rs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE CUDA_VISIBLE_DEVICES=
export PYTHONHASHSEED=1 PYTORCH_JIT=0 RFD2_PINNED=1
export PYTHONPATH=$REF RFD2_REF=$REF

len_arg=(); [ "$LEN" != "-" ] && len_arg=(contigmap.length=$LEN)
sc_ref=(); sc_rs=()
if [ "$EXTRA" = "selfcond" ]; then sc_ref=(inference.str_self_cond=True); sc_rs=(--str-self-cond); fi
# num_designs > 1 exercises the per-design reseed, which nothing has tested.
ND=1
if [ "$EXTRA" = "ndes2" ]; then ND=2; fi

# ---- 1. reference, dumping this protein's own rfi (atom_frames) -------------
t0=$(date +%s.%N)
RFD2_DUMP_RFI=$OUT/rfi.safetensors .venv/bin/python python/run_reference.py \
  --config-name=aa \
  inference.ckpt_path=$REF/rf_diffusion/model_weights/RFD_173.pt \
  inference.input_pdb=$PDB "inference.ligand='$LIGS'" \
  "contigmap.contigs=['$CONTIG']" "${len_arg[@]}" "${sc_ref[@]}" \
  inference.contig_as_guidepost=False inference.num_designs=$ND \
  inference.deterministic=True inference.idealize_sidechain_outputs=False \
  inference.write_trb_indep=False diffuser.T=$T \
  inference.output_prefix=$OUT/ref/design > "$OUT/ref/out.txt" 2> "$OUT/ref/log.txt"
rc_ref=$?; t_ref=$(echo "$(date +%s.%N) - $t0" | bc)
[ $rc_ref -ne 0 ] && { echo "$CASE	REF_FAIL	$(grep -aoE '[A-Za-z]*(Error|Exception)[^\"]{0,90}' "$OUT/ref/log.txt" | tail -1)"; exit 0; }

# ---- 2. ligand sidecar built from THIS run's frames -------------------------
RFD2_ATOM_FRAMES=$OUT/rfi.safetensors .venv/bin/python python/gen_ligand_bonds.py \
  "$PDB" "$LIGS" > "$OUT/ligand.log" 2>&1
rc_l=$?
[ $rc_l -ne 0 ] && { echo "$CASE	LIGAND_FAIL	$(tail -1 "$OUT/ligand.log")"; exit 0; }

# ---- 3. the port ------------------------------------------------------------
len_rs=(); [ "$LEN" != "-" ] && len_rs=(--length "$LEN")
t1=$(date +%s.%N)
./target/release/rfd2 --input-pdb "$PDB" --contigs "$CONTIG" --ligand "$LIGS" \
  --ligand-topology fixtures/ligand/$PROT.safetensors \
  --weights fixtures/weights/model_state_dict.safetensors \
  --igso3 fixtures/noiser/stages.safetensors \
  "${len_rs[@]}" "${sc_rs[@]}" --num-designs $ND --T $T --output-prefix $OUT/rs/design \
  > "$OUT/rs/out.txt" 2>&1
rc_rs=$?; t_rs=$(echo "$(date +%s.%N) - $t1" | bc)
[ $rc_rs -ne 0 ] && { echo "$CASE	PORT_FAIL	$(grep -aoE '(Error|error|refus|unsupported)[^\"]{0,90}' "$OUT/rs/out.txt" | tail -1)"; exit 0; }

# ---- 4. compare -------------------------------------------------------------
for i in $(seq 0 $((ND-1))); do
  fa="$OUT/ref/design_${i}-atomized-bb-False.pdb"; fb="$OUT/rs/design_${i}-atomized-bb-False.pdb"
  [ -f "$fa" ] && [ -f "$fb" ] || { echo "$CASE	MISSING_DESIGN_$i"; continue; }
  tag=$CASE; [ $ND -gt 1 ] && tag="$CASE#d$i"
  .venv/bin/python bench/compare.py "$tag" "$PROT" "$LIGS" "$CONTIG" "$LEN" "$T" "$EXTRA" \
    "$fa" "$fb" "$t_ref" "$t_rs"
done
