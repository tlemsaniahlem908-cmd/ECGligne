from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
import wfdb

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model_defs import (
    load_resnet_hybrid,
    ensure_ct,
    extract_features_one,
    compute_rules,
    apply_norm,
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_ai"
TEMP_DIR.mkdir(exist_ok=True)

A_DIR = BASE_DIR / "model_A" / "model_binary_v10b_simclr_strong"
A1_DIR = BASE_DIR / "model_A1" / "model_binary_A1_norm_vs_mi_sttc_cd"
B3_DIR = BASE_DIR / "model_B3" / "model_binary_modelB_3class"

FOLD = 1

T_A = 0.301
T_A1 = 0.50

CLASSES_B3 = ["MI", "STTC", "CD"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPECTED_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# =========================
# FASTAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ECG NORMALIZATION
# =========================
def normalize_ecg(ecg):
    """
    Input : (5000, 12) in mV
    Output: (5000, 12) z-score per lead
    """
    mean = ecg.mean(axis=0, keepdims=True)
    std = ecg.std(axis=0, keepdims=True) + 1e-8
    return ((ecg - mean) / std).astype(np.float32)


def build_meta_normalized(model_dir, fold=1):
    meta_mean = np.load(model_dir / f"fold{fold}_meta_mean.npy").astype(np.float32)
    meta_std = np.load(model_dir / f"fold{fold}_meta_std.npy").astype(np.float32)

    # population center metadata
    return apply_norm(meta_mean[None, :], meta_mean, meta_std).astype(np.float32)


def prepare_inputs(ecg_mv, model_dir, fold=1):
    """
    ecg_mv shape must be (5000, 12), unit mV.
    """

    ecg_norm = normalize_ecg(ecg_mv)
    ecg_ct = ensure_ct(ecg_norm)  # (12, 5000)

    x = torch.tensor(ecg_ct, dtype=torch.float32).unsqueeze(0)

    feat_raw = extract_features_one(ecg_ct)[None, :]

    feat_mean = np.load(model_dir / f"fold{fold}_feat_mean.npy")
    feat_std = np.load(model_dir / f"fold{fold}_feat_std.npy")
    feat = apply_norm(feat_raw, feat_mean, feat_std)

    ref_feat = np.load(model_dir / "features_pool.npy").astype(np.float32)
    rules = compute_rules(feat_raw, ref_feat)

    meta = build_meta_normalized(model_dir, fold)

    m = torch.tensor(meta, dtype=torch.float32)
    f = torch.tensor(feat, dtype=torch.float32)
    r = torch.tensor(rules, dtype=torch.float32)

    return x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), r.to(DEVICE)


def predict_binary(model, ecg_mv, model_dir):
    x, m, f, r = prepare_inputs(ecg_mv, model_dir, FOLD)

    with torch.no_grad():
        out = torch.sigmoid(model(x, m, f, r)).detach().cpu().numpy()

    return float(out.reshape(-1)[0])


def predict_b3(model, ecg_mv, model_dir):
    x, m, f, r = prepare_inputs(ecg_mv, model_dir, FOLD)

    with torch.no_grad():
        probs = torch.softmax(model(x, m, f, r), dim=1).detach().cpu().numpy()[0]

    return probs.astype(float)


def read_wfdb_ecg(record_base_path: Path):
    """
    record_base_path is without extension:
    temp_ai/xxxx/record
    It reads record.hea + record.dat.
    """

    record = wfdb.rdrecord(str(record_base_path))

    ecg_mv = record.p_signal
    fs = int(record.fs)
    lead_names = list(record.sig_name)

    if ecg_mv is None:
        raise ValueError("Could not read ECG p_signal from WFDB files")

    if fs != 500:
        raise ValueError(f"Expected ECG sampling rate 500 Hz, got {fs} Hz")

    ecg_mv = np.nan_to_num(ecg_mv, nan=0.0).astype(np.float32)

    if ecg_mv.shape[0] < 5000:
        raise ValueError(f"Expected at least 5000 samples, got {ecg_mv.shape[0]}")

    if ecg_mv.shape[1] != 12:
        raise ValueError(f"Expected 12 leads, got {ecg_mv.shape[1]}")

    # Take exact first 10 seconds
    ecg_mv = ecg_mv[:5000, :]

    # Reorder leads if names are available
    upper_names = [x.upper() for x in lead_names]

    if all(lead in upper_names for lead in EXPECTED_LEADS):
        indexes = [upper_names.index(lead) for lead in EXPECTED_LEADS]
        ecg_mv = ecg_mv[:, indexes]
        lead_names = EXPECTED_LEADS

    return ecg_mv, fs, lead_names


# =========================
# LOAD MODELS ONCE
# =========================
print("====================================")
print("AI ECG SERVER STARTING")
print("Device:", DEVICE)
print("Loading models...")
print("====================================")

model_A = load_resnet_hybrid(
    str(A_DIR / f"fold{FOLD}_resnet_hybrid_best.pt"),
    n_outputs=1,
).to(DEVICE)

model_A1 = load_resnet_hybrid(
    str(A1_DIR / f"fold{FOLD}_resnet_hybrid_best.pt"),
    n_outputs=1,
).to(DEVICE)

model_B3 = load_resnet_hybrid(
    str(B3_DIR / f"fold{FOLD}_resnet_hybrid_best.pt"),
    n_outputs=3,
).to(DEVICE)

model_A.eval()
model_A1.eval()
model_B3.eval()

print("Models loaded successfully.")


# =========================
# API ENDPOINT
# =========================
@app.post("/predict_ecg")
async def predict_ecg(
    hea_file: UploadFile = File(...),
    dat_file: UploadFile = File(...),
):
    request_id = str(uuid.uuid4())
    work_dir = TEMP_DIR / request_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        hea_path = work_dir / "record.hea"
        dat_path = work_dir / "record.dat"

        with hea_path.open("wb") as f:
            shutil.copyfileobj(hea_file.file, f)

        with dat_path.open("wb") as f:
            shutil.copyfileobj(dat_file.file, f)

        # Ensure header points to record.dat
        hea_text = hea_path.read_text(errors="ignore")
        lines = hea_text.splitlines()

        if not lines:
            raise ValueError("Empty .hea file")

        first_parts = lines[0].split()
        old_record_name = first_parts[0]
        first_parts[0] = "record"
        lines[0] = " ".join(first_parts)

        fixed_lines = [lines[0]]

        for line in lines[1:]:
            fixed_line = line.replace(f"{old_record_name}.dat", "record.dat")
            fixed_line = fixed_line.replace(old_record_name, "record")
            fixed_lines.append(fixed_line)

        hea_path.write_text("\n".join(fixed_lines) + "\n")

        ecg_mv, fs, lead_names = read_wfdb_ecg(work_dir / "record")

        pA = predict_binary(model_A, ecg_mv, A_DIR)
        pA1 = predict_binary(model_A1, ecg_mv, A1_DIR)
        probs_B3 = predict_b3(model_B3, ecg_mv, B3_DIR)

        b3_dict = {
            cls: round(float(prob), 6)
            for cls, prob in zip(CLASSES_B3, probs_B3)
        }

        if pA < T_A:
            final = "NORMAL"
        elif pA1 < T_A1:
            final = "NORMAL"
        else:
            final = CLASSES_B3[int(np.argmax(probs_B3))]

        return {
            "success": True,
            "prediction": final,
            "final": final,
            "pA": round(float(pA), 6),
            "pA1": round(float(pA1), 6),
            "B3": b3_dict,
            "fs": fs,
            "shape": list(ecg_mv.shape),
            "leads": lead_names,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    finally:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass