# =============================================================
# Model A1 — label preparation
# Binary task: NORM (0) vs ABNORMAL = MI + STTC + CD (1)
# (A1 adds the CD class to the abnormal group compared with Model A.)
# Recordings that are neither NORM nor {MI,STTC,CD} are ignored (-1).
# Reuses the shared PTB-XL arrays (ECG / meta / splits) unchanged.
# =============================================================
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

BASE = "/home/jupyter/ptbxl_processed"
NEW  = "/home/jupyter/ptbxl_processed_A1_norm_vs_mi_sttc_cd"

Path(NEW).mkdir(parents=True, exist_ok=True)

detail = pd.read_csv(f"{BASE}/ptbxl_labels_detail.csv")

y_new = np.full(len(detail), -1, dtype=np.int8)

for i, row in detail.iterrows():
    if row["has_norm"]:
        y_new[i] = 0
    else:
        p = str(row["pathologies"])
        if ("MI" in p) or ("STTC" in p) or ("CD" in p):
            y_new[i] = 1
        else:
            y_new[i] = -1

print("A1 LABELS:")
print("NORM     :", int((y_new == 0).sum()))
print("ABNORMAL :", int((y_new == 1).sum()))
print("IGNORED  :", int((y_new == -1).sum()))
print("TOTAL    :", int((y_new != -1).sum()))
print("BALANCE  :", round((y_new == 1).sum() / (y_new != -1).sum() * 100, 1), "% ABNORMAL")

np.save(f"{NEW}/ptbxl_labels.npy", y_new)

for f in [
    "ptbxl_ecg.npy",
    "ptbxl_meta.npy",
    "ptbxl_splits.npz",
    "ptbxl_labels_detail.csv",
    "ptbxl_meta_cols.txt",
    "ptbxl_ecg_ids.npy"
]:
    src = f"{BASE}/{f}"
    if Path(src).exists():
        shutil.copy(src, f"{NEW}/{f}")

print("DONE ✅")
print("New dataset:", NEW)
