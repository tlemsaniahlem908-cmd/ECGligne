# =============================================================
# Model A — label preparation
# Binary task: NORM (0) vs ABNORMAL = MI + STTC (1)
# Recordings that are neither NORM nor {MI,STTC} are ignored (-1).
# Reuses the shared PTB-XL arrays (ECG / meta / splits) unchanged.
# =============================================================
import pandas as pd
import numpy as np
from pathlib import Path

BASE = "/home/jupyter/ptbxl_processed"
NEW  = "/home/jupyter/ptbxl_processed_modelA_strong"
Path(NEW).mkdir(parents=True, exist_ok=True)

# load
detail = pd.read_csv(f"{BASE}/ptbxl_labels_detail.csv")

# new labels
y_new = np.full(len(detail), -1, dtype=np.int8)

for i, row in detail.iterrows():
    if row["has_norm"]:
        y_new[i] = 0
    else:
        p = str(row["pathologies"])

        if ("MI" in p) or ("STTC" in p):
            y_new[i] = 1
        else:
            y_new[i] = -1

# stats
print("NEW LABELS:")
print(np.unique(y_new, return_counts=True))

# save
np.save(f"{NEW}/ptbxl_labels.npy", y_new)

# copy other files (no change)
import shutil

for f in ["ptbxl_ecg.npy", "ptbxl_meta.npy", "ptbxl_splits.npz"]:
    shutil.copy(f"{BASE}/{f}", f"{NEW}/{f}")

print("DONE ✅")
