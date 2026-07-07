import numpy as np
import torch

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
LIVE_ECG_PATH = r"..\ecg12_500hz_esp32\live_ecg_10s.npy"

A_DIR  = r"model_A\model_binary_v10b_simclr_strong"
A1_DIR = r"model_A1\model_binary_A1_norm_vs_mi_sttc_cd"
B3_DIR = r"model_B3\model_binary_modelB_3class"

# ── ENSEMBLE des 5 folds (avant : un seul FOLD = 1) ──
FOLDS = [1, 2, 3, 4, 5]

T_A  = 0.301
T_A1 = 0.50

CLASSES_B3 = ["MI", "STTC", "CD"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ECG z-score normalization
# =========================
def normalize_ecg(ecg):
    """Z-score per lead. Input (5000, 12). Output (5000, 12)."""
    mean = ecg.mean(axis=0, keepdims=True)
    std  = ecg.std(axis=0, keepdims=True) + 1e-8
    return ((ecg - mean) / std).astype(np.float32)


# =========================
# Meta: population center (par fold)
# =========================
def build_meta_normalized(model_dir, fold):
    meta_mean = np.load(f"{model_dir}/fold{fold}_meta_mean.npy").astype(np.float32)
    meta_std  = np.load(f"{model_dir}/fold{fold}_meta_std.npy").astype(np.float32)
    # population mean normalizes to exactly 0 = in-distribution center
    return apply_norm(meta_mean[None, :], meta_mean, meta_std).astype(np.float32)


# =========================
# PREPARE INPUTS (par fold)
# =========================
def prepare_inputs(ecg_mv, model_dir, fold):
    """
    ecg_mv : (5000, 12) float32 in mV
    Z-score normalise pour le signal ET les features (comme a l'entrainement).
    """
    # Z-score normalize
    ecg_norm = normalize_ecg(ecg_mv)          # (5000, 12)
    ecg_ct   = ensure_ct(ecg_norm)            # (12, 5000)

    # Model input tensor
    x = torch.tensor(ecg_ct, dtype=torch.float32).unsqueeze(0)  # (1, 12, 5000)

    # Features on z-score signal (same as training)
    feat_raw = extract_features_one(ecg_ct)[None, :]            # (1, 16)

    feat_mean = np.load(f"{model_dir}/fold{fold}_feat_mean.npy")
    feat_std  = np.load(f"{model_dir}/fold{fold}_feat_std.npy")
    feat      = apply_norm(feat_raw, feat_mean, feat_std)

    # Clinical rules on z-score features
    ref_feat = np.load(f"{model_dir}/features_pool.npy").astype(np.float32)
    rules    = compute_rules(feat_raw, ref_feat)

    # Meta (patient moyen)
    meta = build_meta_normalized(model_dir, fold)

    m = torch.tensor(meta,  dtype=torch.float32)
    f = torch.tensor(feat,  dtype=torch.float32)
    r = torch.tensor(rules, dtype=torch.float32)

    return x.to(DEVICE), m.to(DEVICE), f.to(DEVICE), r.to(DEVICE)


# =========================
# INFERENCE — ENSEMBLE des 5 folds
# =========================
def predict_binary_ensemble(ecg_mv, model_dir):
    """Moyenne des probas des 5 folds (NORM vs ABNORMAL)."""
    probs = []
    for fold in FOLDS:
        model = load_resnet_hybrid(
            f"{model_dir}/fold{fold}_resnet_hybrid_best.pt", n_outputs=1
        ).to(DEVICE)
        model.eval()
        x, m, f, r = prepare_inputs(ecg_mv, model_dir, fold)
        with torch.no_grad():
            p = torch.sigmoid(model(x, m, f, r)).cpu().numpy()[0]
        probs.append(float(p))
    return float(np.mean(probs))


def predict_b3_ensemble(ecg_mv, model_dir):
    """Moyenne des probas softmax des 5 folds (MI / STTC / CD)."""
    probs = []
    for fold in FOLDS:
        model = load_resnet_hybrid(
            f"{model_dir}/fold{fold}_resnet_hybrid_best.pt", n_outputs=3
        ).to(DEVICE)
        model.eval()
        x, m, f, r = prepare_inputs(ecg_mv, model_dir, fold)
        with torch.no_grad():
            p = torch.softmax(model(x, m, f, r), dim=1).cpu().numpy()[0]
        probs.append(p)
    return np.mean(np.stack(probs, axis=0), axis=0)


# =========================
# MAIN
# =========================
print("Device:", DEVICE)

ecg_mv = np.load(LIVE_ECG_PATH).astype(np.float32)
print(f"Loaded ECG  : shape={ecg_mv.shape}  unit=mV")
print(f"              min={ecg_mv.min():.4f}  max={ecg_mv.max():.4f}")
print(f"Ensemble    : {len(FOLDS)} folds {FOLDS}")

# Sanity check features on z-score (fold 1 reference)
ecg_norm_check = normalize_ecg(ecg_mv)
ecg_ct_check   = ensure_ct(ecg_norm_check)
feat_check     = extract_features_one(ecg_ct_check)
print(f"\nFeature check (z-score signal):")
print(f"  HR        = {feat_check[0]:.1f} bpm")
print(f"  energy    = {feat_check[5]:.4f}")
print(f"  n_peaks   = {feat_check[15]:.0f}")

# Run inference (moyenne des 5 folds)
pA       = predict_binary_ensemble(ecg_mv, A_DIR)
pA1      = predict_binary_ensemble(ecg_mv, A1_DIR)
probs_B3 = predict_b3_ensemble(ecg_mv, B3_DIR)

print("\n==============================")
print("MODEL OUTPUTS (moyenne des 5 folds)")
print("==============================")
print(f"pA   NORM vs ABNORMAL  : {pA:.4f}   threshold={T_A}")
print(f"pA1  NORM vs MI/STTC/CD: {pA1:.4f}   threshold={T_A1}")
print(f"B3   probs             : { {c: round(float(p),4) for c,p in zip(CLASSES_B3, probs_B3)} }")

# Decision (cascade — inchangee)
if pA < T_A:
    final = "NORMAL"
elif pA1 < T_A1:
    final = "NORMAL"
else:
    final = CLASSES_B3[int(np.argmax(probs_B3))]

print("\n==============================")
print(f"FINAL ECG RESULT : {final}")
print("==============================")
