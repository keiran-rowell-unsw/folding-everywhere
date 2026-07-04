"""Compare the fp32-reference PyTorch coords vs the original bf16-reference coords
(same seed) for one protein — quantifies how much the bf16 atom-attention
optimization moves the predicted structure. Kabsch-aligned RMSD."""
import sys, os, numpy as np
name = sys.argv[1] if len(sys.argv) > 1 else "flgM97"
FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
fp32 = np.load(os.path.join(FIX, f"smp_{name}_coords.npy"))      # fp32-reference (regenerated)
bf16 = np.load(os.path.join(FIX, f"{name}_coords_bf16.npy"))     # bf16-reference (backup)

def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Pa = Pc @ R.T
    diff = Pa - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P))), float(np.abs(diff).max())

r, m = kabsch_rmsd(bf16, fp32)
print(f"=== PyTorch bf16 vs fp32 reference ({name}) ===")
print(f"Kabsch-aligned RMSD = {r:.4f} A,  max atom deviation = {m:.4f} A")
print(f"raw (no realign) max|diff| = {np.abs(fp32 - bf16).max():.4f} A")
