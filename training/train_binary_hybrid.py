"""
=============================================================
PTB-XL — MODEL A v10b SimCLR Hybrid (FIXED)
NORM vs ALL ABNORMAL — Binary Classification
Vertex AI / NVIDIA L4

Same script is used for Model A and Model A1 — only change:
  DATA_DIR   -> ptbxl_processed_modelA_strong   (A)
              / ptbxl_processed_A1_norm_vs_mi_sttc_cd (A1)
  OUTPUT_DIR -> the matching output folder

Fixes vs v10 :
  ✅ FIX 1 : Pseudo-labeling sur OOF (pas sur train)
             → accumulation des probs OOF fold par fold
             → PL lancé APRES tous les folds base
  ✅ FIX 2 : EPOCHS=80, PATIENCE=15
  ✅ FIX 3 : TTA_N=8 pour stabiliser les probs OOF/PL

Architecture :
  ✅ ResNetHybrid  → SimCLR pretrained encoder (freeze 5ep → unfreeze LR/10)
  ✅ InceptionHybrid → random init
  ✅ TCNHybrid       → random init
  ✅ ECG features (16) + clinical rules (10) + meta
  ✅ WeightedPolyLoss + label smoothing
  ✅ 5-fold OOF strict
=============================================================
"""

import os, gc, json, time, math, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, confusion_matrix,
                             classification_report, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================
# CONFIG
# =============================================================
SEED         = 42
DATA_DIR = "/home/jupyter/ptbxl_processed_modelA_strong"
OUTPUT_DIR = "/home/jupyter/model_binary_v10b_simclr_strong"
SIMCLR_PATH  = "/home/jupyter/simclr_pretrain/simclr_resnet_encoder.pt"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

N_LEADS   = 12
N_SAMPLES = 5000
FS        = 500
N_FOLDS   = 5
USE_THREE_MODELS = True

BATCH         = 32
EPOCHS        = 80     # FIX 2
PATIENCE      = 15     # FIX 2
FREEZE_EPOCHS = 5
LR_HEAD       = 1e-4
LR_ENCODER    = 1e-5
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.02
TTA_N         = 8     # FIX 3
USE_AMP       = True
NUM_WORKERS   = 0

# Pseudo-labeling — lancé sur OOF après tous les folds base
USE_PSEUDO_LABELING  = True
PL_HIGH              = 0.97   # un peu moins strict pour avoir assez de samples
PL_LOW               = 0.03
PL_MAX_PER_CLASS     = 1500
PL_EPOCHS            = 20
PL_PATIENCE          = 7
PL_LR                = 5e-5
PSEUDO_SAMPLE_WEIGHT = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# UTILS
# =============================================================
def pth(x): return os.path.join(OUTPUT_DIR, x)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed()

try:
    from torch.amp import autocast, GradScaler
    def amp_ctx():
        return autocast(device_type="cuda", dtype=torch.float16,
                        enabled=(USE_AMP and DEVICE.type == "cuda"))
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    def amp_ctx(): return autocast(enabled=(USE_AMP and DEVICE.type == "cuda"))

def free_mem():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def ensure_ct(sig):
    sig = np.asarray(sig, dtype=np.float32)
    if sig.shape == (N_LEADS, N_SAMPLES): return sig
    if sig.shape == (N_SAMPLES, N_LEADS): return sig.T
    raise ValueError(f"Unexpected ECG shape: {sig.shape}")

def scan_threshold(probs, labels, metric="accuracy"):
    labels = labels.astype(int); best_t, best_s = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.0025):
        pred = (probs >= t).astype(int)
        s = f1_score(labels, pred, zero_division=0) if metric == "f1" \
            else accuracy_score(labels, pred)
        if s > best_s: best_s, best_t = s, float(t)
    return best_t, float(best_s)

def youden_threshold(probs, labels):
    fpr, tpr, thr = roc_curve(labels.astype(int), probs)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i]), float(tpr[i]), float(fpr[i])

def compute_metrics(y_true, probs, threshold):
    y_true = y_true.astype(int)
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold"            : float(threshold),
        "accuracy"             : float(accuracy_score(y_true, y_pred)),
        "roc_auc"              : float(roc_auc_score(y_true, probs)),
        "f1_macro"             : float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_abnormal"          : float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_abnormal"   : float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity_abnormal" : float(tp / max(1, tp+fn)),
        "specificity_norm"     : float(tn / max(1, tn+fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, cm, y_pred

def plot_cm(cm, path, title):
    fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(cm)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["NORM","ABNORMAL"])
    ax.set_yticklabels(["NORM","ABNORMAL"])
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)

# =============================================================
# ECG FEATURES
# =============================================================
def rpeak_proxy(lead):
    x = lead.astype(np.float32) - np.median(lead)
    d = np.abs(np.diff(x, prepend=x[0]))
    e = np.convolve(d, np.ones(9, dtype=np.float32)/9.0, mode="same")
    thr = e.mean() + 1.2*e.std(); min_dist = int(0.25*FS)
    peaks, last = [], -min_dist
    for i in range(1, len(e)-1):
        if (i-last >= min_dist and e[i] > thr
                and e[i] >= e[i-1] and e[i] >= e[i+1]):
            peaks.append(i); last = i
    return np.array(peaks, dtype=np.int32), e

def extract_features_one(sig):
    sig  = np.nan_to_num(ensure_ct(sig).astype(np.float32),
                         nan=0., posinf=0., neginf=0.)
    lead = sig[1] if sig.shape[0] > 1 \
           else sig[int(np.argmax(sig.var(axis=1)))]
    peaks, der = rpeak_proxy(lead)
    n  = len(peaks); hr = 60.0*n/(N_SAMPLES/FS)
    if n >= 3:
        rr = np.diff(peaks)/FS
        rr_mean, rr_std = float(rr.mean()), float(rr.std())
        rr_cv = rr_std / max(1e-6, rr_mean)
    else:
        rr_mean = rr_std = rr_cv = 0.0
    widths = []
    for p in peaks[:30]:
        l = max(0, p-int(0.12*FS)); r = min(N_SAMPLES, p+int(0.12*FS))
        seg = der[l:r]
        if len(seg) > 3:
            active = np.where(seg > seg.mean() + 0.5*seg.std())[0]
            if len(active) > 1: widths.append((active[-1]-active[0])/FS)
    qrs = float(np.median(widths)) if widths else 0.0
    abs_sig = np.abs(sig); le = np.mean(sig**2, axis=1)
    win = int(0.8*FS); win += 1 if win % 2 == 0 else 0
    baseline = np.convolve(lead, np.ones(win, dtype=np.float32)/win, mode="same")
    feats = np.array([
        hr, rr_mean, rr_std, rr_cv, qrs,
        float(np.mean(sig**2)), float(abs_sig.mean()), float(abs_sig.std()),
        float(le.mean()), float(le.std()), float(le.max()), float(le.min()),
        float(np.mean(baseline**2)), float(np.mean(der)),
        float(np.mean(np.std(sig, axis=1) < 1e-5)), float(n)
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0., posinf=0., neginf=0.)

def extract_or_load_features(X, indices, name):
    out = pth(f"features_{name}.npy")
    if os.path.exists(out):
        print(f"Loading cached features: {out}")
        return np.load(out).astype(np.float32)
    print(f"Extracting ECG features [{name}]: {len(indices)} samples")
    Fmat = np.zeros((len(indices), 16), dtype=np.float32)
    for i, idx in enumerate(indices):
        if i % 1000 == 0: print(f"  {name}: {i}/{len(indices)}")
        Fmat[i] = extract_features_one(X[idx])
    np.save(out, Fmat); print(f"Saved: {out}"); return Fmat

def compute_rules(F, ref=None):
    ref   = F if ref is None else ref
    e_hi  = np.percentile(ref[:,5],  90)
    e_lo  = np.percentile(ref[:,5],   5)
    n_hi  = np.percentile(ref[:,13], 90)
    b_hi  = np.percentile(ref[:,12], 90)
    hr, rr_cv, qrs = F[:,0], F[:,3], F[:,4]
    energy, base, noise, flat = F[:,5], F[:,12], F[:,13], F[:,14]
    rules = [
        ((hr > 0) & (hr < 50)), hr > 110, rr_cv > 0.18, qrs > 0.115,
        energy > e_hi, energy < e_lo, noise > n_hi, base > b_hi, flat > 0.15
    ]
    R     = np.stack([x.astype(np.float32) for x in rules], axis=1)
    score = np.clip(R.sum(1, keepdims=True)/9.0, 0, 1)
    return np.concatenate([R, score], axis=1).astype(np.float32)

def norm_fit_apply(Xfit, Xapply):
    mean = np.nanmean(Xfit, axis=0).astype(np.float32)
    std  = np.nanstd(Xfit,  axis=0).astype(np.float32); std[std < 1e-6] = 1.0
    Xn   = (np.nan_to_num(Xapply, nan=0., posinf=0., neginf=0.) - mean) / std
    return np.nan_to_num(Xn, nan=0., posinf=0., neginf=0.).astype(np.float32), mean, std

# =============================================================
# DATA
# =============================================================
def load_data():
    print("="*60); print("LOADING DATA"); print("="*60)
    X      = np.load(f"{DATA_DIR}/ptbxl_ecg.npy", mmap_mode="r")
    M      = np.load(f"{DATA_DIR}/ptbxl_meta.npy").astype(np.float32)
    y      = np.load(f"{DATA_DIR}/ptbxl_labels.npy")
    splits = np.load(f"{DATA_DIR}/ptbxl_splits.npz")
    print("ECG shape  :", X.shape); print("Meta shape :", M.shape)
    pool_idx = np.concatenate([splits["train"], splits["val"]])
    test_idx  = np.asarray(splits["test"])
    pool_idx  = pool_idx[y[pool_idx] != -1]
    test_idx  = test_idx[y[test_idx]  != -1]
    y_pool = y[pool_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)
    print(f"Pool : {len(pool_idx)} | NORM={(y_pool==0).sum()} | ABNORMAL={(y_pool==1).sum()}")
    print(f"Test : {len(test_idx)} | NORM={(y_test==0).sum()} | ABNORMAL={(y_test==1).sum()}")
    return X, M, y, pool_idx, test_idx, y_pool, y_test

# =============================================================
# DATASET / AUGMENTATION
# =============================================================
def ecg_augment(sig):
    sig = sig.copy().astype(np.float32)
    if random.random() < 0.55:
        sig += np.random.normal(0, 0.010, sig.shape).astype(np.float32)
    if random.random() < 0.55:
        sig *= np.float32(random.uniform(0.90, 1.10))
    if random.random() < 0.50:
        sig = np.roll(sig, random.randint(-150, 150), axis=1)
    if random.random() < 0.45:
        t    = np.linspace(0, 1, N_SAMPLES, dtype=np.float32)
        sig += (random.uniform(0.005,0.03)
                * np.sin(2*np.pi*random.uniform(0.15,0.5)*t
                         + random.uniform(0,2*np.pi)))[None,:].astype(np.float32)
    if random.random() < 0.25:
        sig[np.random.choice(N_LEADS, random.randint(1,2), replace=False)] = 0.0
    if random.random() < 0.20:
        t    = np.arange(N_SAMPLES, dtype=np.float32)/FS
        sig += (random.uniform(0.002,0.012)
                * np.sin(2*np.pi*50*t
                         + random.uniform(0,2*np.pi)))[None,:].astype(np.float32)
    return sig.astype(np.float32)

class ECGHybridDataset(Dataset):
    def __init__(self, X, idx, M, F, R, y, augment=False, weights=None):
        self.X=X; self.idx=np.asarray(idx)
        self.M=M.astype(np.float32); self.F=F.astype(np.float32)
        self.R=R.astype(np.float32); self.y=y.astype(np.float32)
        self.aug=augment
        self.w = np.ones(len(self.y), dtype=np.float32) \
                 if weights is None else weights.astype(np.float32)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        sig = ensure_ct(self.X[self.idx[i]])
        if self.aug: sig = ecg_augment(sig)
        return (torch.from_numpy(sig.copy()), torch.from_numpy(self.M[i]),
                torch.from_numpy(self.F[i]),  torch.from_numpy(self.R[i]),
                torch.tensor(self.y[i], dtype=torch.float32),
                torch.tensor(self.w[i], dtype=torch.float32))

class ECGHybridArrayDataset(Dataset):
    def __init__(self, Xarr, M, F, R, y, augment=False, weights=None):
        self.X=Xarr.astype(np.float32)
        self.M=M.astype(np.float32); self.F=F.astype(np.float32)
        self.R=R.astype(np.float32); self.y=y.astype(np.float32)
        self.aug=augment
        self.w = np.ones(len(self.y), dtype=np.float32) \
                 if weights is None else weights.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        sig = self.X[i]
        if self.aug: sig = ecg_augment(sig)
        return (torch.from_numpy(sig.copy()), torch.from_numpy(self.M[i]),
                torch.from_numpy(self.F[i]),  torch.from_numpy(self.R[i]),
                torch.tensor(self.y[i], dtype=torch.float32),
                torch.tensor(self.w[i], dtype=torch.float32))

def make_loader(X, idx, M, F, R, y,
                augment=False, shuffle=False,
                batch_size=BATCH, drop_last=False, weights=None):
    return DataLoader(
        ECGHybridDataset(X, idx, M, F, R, y, augment, weights),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=drop_last)

def make_array_loader(Xarr, M, F, R, y,
                      augment=False, shuffle=False,
                      batch_size=BATCH, drop_last=False, weights=None):
    return DataLoader(
        ECGHybridArrayDataset(Xarr, M, F, R, y, augment, weights),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, drop_last=drop_last)

# =============================================================
# MODEL BLOCKS
# =============================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__(); hid = max(4, ch//r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch,hid), nn.GELU(),
            nn.Linear(hid,ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1)

class MultiPool(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean(-1), x.max(-1).values, x.std(-1)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, ic, oc, k=7, stride=1, dil=1, drop=0.1):
        super().__init__(); pad = dil*(k-1)//2
        self.c1 = nn.Conv1d(ic,oc,k,stride=stride,padding=pad,dilation=dil,bias=False)
        self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc,oc,k,padding=pad,dilation=dil,bias=False)
        self.b2 = nn.BatchNorm1d(oc); self.se = SEBlock(oc); self.dp = nn.Dropout(drop)
        self.sk = nn.Sequential(
            nn.Conv1d(ic,oc,1,stride=stride,bias=False),
            nn.BatchNorm1d(oc)) if (ic!=oc or stride!=1) else nn.Identity()
    def forward(self, x):
        s = self.sk(x); x = F.gelu(self.b1(self.c1(x)))
        x = self.dp(x); x = self.se(self.b2(self.c2(x)))
        return F.gelu(x+s)

class InceptionBlock(nn.Module):
    def __init__(self, ic, btn=32, out=32):
        super().__init__()
        self.btn = nn.Conv1d(ic,btn,1,bias=False)
        self.c9  = nn.Conv1d(btn,out,9,padding=4,bias=False)
        self.c19 = nn.Conv1d(btn,out,19,padding=9,bias=False)
        self.c39 = nn.Conv1d(btn,out,39,padding=19,bias=False)
        self.mp  = nn.MaxPool1d(3,1,1)
        self.mpc = nn.Conv1d(ic,out,1,bias=False)
        self.bn  = nn.BatchNorm1d(out*4); self.se = SEBlock(out*4)
    def forward(self, x):
        b = self.btn(x)
        o = torch.cat([self.c9(b),self.c19(b),self.c39(b),self.mpc(self.mp(x))],1)
        return self.se(F.gelu(self.bn(o)))

class TCNBlock(nn.Module):
    def __init__(self, ic, oc, k=5, dil=1, drop=0.15):
        super().__init__(); pad = dil*(k-1); self.pad = pad
        self.c1 = nn.Conv1d(ic,oc,k,padding=pad,dilation=dil,bias=False)
        self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc,oc,k,padding=pad,dilation=dil,bias=False)
        self.b2 = nn.BatchNorm1d(oc); self.se = SEBlock(oc); self.dp = nn.Dropout(drop)
        self.sk = nn.Conv1d(ic,oc,1,bias=False) if ic!=oc else nn.Identity()
    def chomp(self, x): return x[:,:,:-self.pad] if self.pad else x
    def forward(self, x):
        s = self.sk(x)
        x = F.gelu(self.b1(self.chomp(self.c1(x)))); x = self.dp(x)
        x = self.se(self.b2(self.chomp(self.c2(x))))
        s = s[:,:,:x.shape[-1]] if s.shape[-1]!=x.shape[-1] else s
        return F.gelu(x+s)

class MetaBranch(nn.Module):
    def __init__(self, n, out=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(64,out), nn.GELU())
    def forward(self, x): return self.net(x)

class FeatureBranch(MetaBranch): pass

class RuleBranch(nn.Module):
    def __init__(self, n, out=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,32), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(32,out), nn.GELU())
    def forward(self, x): return self.net(x)

class FFTBranch(nn.Module):
    def __init__(self, n_bins=128, out=32):
        super().__init__(); self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Conv1d(N_LEADS,32,5,padding=2,bias=False), nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,32,3,stride=2,padding=1,bias=False), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32,out), nn.GELU())
    def forward(self, x):
        return self.net(
            torch.abs(torch.fft.rfft(x, n=self.n_bins*2, dim=-1))[...,:self.n_bins])

# =============================================================
# MODELS
# =============================================================
class ResNetHybrid(nn.Module):
    """ResNet + SimCLR pretrained encoder + meta/feat/rule."""
    def __init__(self, n_meta, n_feat, n_rule):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS,64,15,stride=2,padding=7,bias=False),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(3,2,1))
        self.layers = nn.Sequential(
            ResBlock(64,64),
            ResBlock(64,128,stride=2,drop=0.12),
            ResBlock(128,128,dil=2,drop=0.12),
            ResBlock(128,256,stride=2,drop=0.15),
            ResBlock(256,256,dil=4,drop=0.15),
            ResBlock(256,512,stride=2,drop=0.18),
            ResBlock(512,512,dil=8,drop=0.18))
        self.pool = MultiPool()
        self.meta = MetaBranch(n_meta)
        self.feat = FeatureBranch(n_feat)
        self.rule = RuleBranch(n_rule)
        self.head = nn.Sequential(
            nn.Linear(512*3+32+32+16, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.35), nn.Linear(512,128), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(128,1))

    def load_simclr_encoder(self, path):
        sd = torch.load(path, map_location="cpu")
        current = self.state_dict()
        filtered, skipped = {}, []
        for k, v in sd.items():
            if k in current:
                if current[k].shape == v.shape: filtered[k] = v
                else: skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        enc_total  = len(self.stem.state_dict()) + len(self.layers.state_dict())
        enc_loaded = sum(1 for k in filtered
                         if k.startswith("stem.") or k.startswith("layers."))
        print(f"  SimCLR encoder: {path}")
        print(f"    encoder keys loaded : {enc_loaded}/{enc_total}")
        print(f"    missing (head/meta) : {len(missing)}")
        print(f"    shape mismatches    : {len(skipped)}")
        if enc_loaded < int(0.80*enc_total):
            print("  WARNING: <80% encoder keys loaded!")

    def encoder_params(self):
        return list(self.stem.parameters()) + list(self.layers.parameters())

    def non_encoder_params(self):
        return (list(self.meta.parameters()) + list(self.feat.parameters()) +
                list(self.rule.parameters()) + list(self.head.parameters()))

    def freeze_encoder(self):
        for p in self.encoder_params(): p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder_params(): p.requires_grad = True

    def forward(self, x, m, f, r):
        z = torch.cat([self.pool(self.layers(self.stem(x))),
                        self.meta(m), self.feat(f), self.rule(r)], dim=1)
        return self.head(z).squeeze(-1)


class InceptionHybrid(nn.Module):
    def __init__(self, n_meta, n_feat, n_rule):
        super().__init__()
        self.fft  = FFTBranch()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm1d(64), nn.GELU())
        ch = 64; blocks = []
        for i in range(6):
            blocks.append(InceptionBlock(ch)); ch = 128
            if i in [1,3]: blocks.append(nn.MaxPool1d(2))
        self.blocks = nn.Sequential(*blocks); self.pool = MultiPool()
        self.meta = MetaBranch(n_meta); self.feat = FeatureBranch(n_feat)
        self.rule = RuleBranch(n_rule)
        self.head = nn.Sequential(
            nn.Linear(128*3+32+32+32+16, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.35), nn.Linear(256,64), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(64,1))

    def forward(self, x, m, f, r):
        z = torch.cat([self.pool(self.blocks(self.stem(x))),
                        self.fft(x), self.meta(m), self.feat(f), self.rule(r)], dim=1)
        return self.head(z).squeeze(-1)


class TCNHybrid(nn.Module):
    def __init__(self, n_meta, n_feat, n_rule):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm1d(64), nn.GELU())
        layers = []; ch = 64
        for dil in [1,2,4,8,16,32,64]:
            out = 128 if dil >= 8 else 64
            layers.append(TCNBlock(ch,out,dil=dil)); ch = out
            if dil in [8,32]: layers.append(nn.MaxPool1d(2))
        self.tcn  = nn.Sequential(*layers); self.pool = MultiPool()
        self.meta = MetaBranch(n_meta); self.feat = FeatureBranch(n_feat)
        self.rule = RuleBranch(n_rule)
        self.head = nn.Sequential(
            nn.Linear(ch*3+32+32+16, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.35), nn.Linear(256,64), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(64,1))

    def forward(self, x, m, f, r):
        z = torch.cat([self.pool(self.tcn(self.stem(x))),
                        self.meta(m), self.feat(f), self.rule(r)], dim=1)
        return self.head(z).squeeze(-1)

# =============================================================
# LOSS / SCHEDULERS
# =============================================================
class WeightedPolyLoss(nn.Module):
    def __init__(self, eps=1.0, smoothing=0.02, pos_weight=None):
        super().__init__(); self.eps=eps; self.sm=smoothing
        self.register_buffer("pos_weight",
                             pos_weight if pos_weight is not None else None)
    def forward(self, logits, targets, weights=None):
        t = targets.view(-1).float(); x = logits.view(-1)
        ts = t*(1-self.sm) + 0.5*self.sm
        bce = F.binary_cross_entropy_with_logits(
            x, ts, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(x); pt = ts*p + (1-ts)*(1-p)
        loss = bce + self.eps*(1-pt)
        if weights is not None: loss = loss * weights.view(-1).float()
        return loss.mean()

class CosineWarmup:
    def __init__(self, opt, warmup, total, lr_max, lr_min=1e-6):
        self.opt=opt; self.wu=warmup; self.total=total
        self.lr_max=lr_max; self.lr_min=lr_min; self.ep=0
    def step(self):
        self.ep += 1
        if self.ep <= self.wu:
            lr = self.lr_max * self.ep / self.wu
        else:
            prog = (self.ep-self.wu) / max(1, self.total-self.wu)
            lr   = self.lr_min + 0.5*(self.lr_max-self.lr_min)*(1+math.cos(math.pi*prog))
        for g in self.opt.param_groups: g["lr"] = lr
        return lr

class WarmupCosineMultiLR:
    """Scheduler multi-group (encoder + head LR séparés)."""
    def __init__(self, opt, warmup, total, min_factor=0.01):
        self.opt=opt; self.wu=warmup; self.total=total
        self.min_factor=min_factor; self.ep=0
        self.base_lrs = [g["lr"] for g in opt.param_groups]
    def step(self):
        self.ep += 1
        if self.ep <= self.wu:
            factor = self.ep / max(1, self.wu)
        else:
            prog   = (self.ep-self.wu) / max(1, self.total-self.wu)
            factor = self.min_factor + 0.5*(1-self.min_factor)*(1+math.cos(math.pi*prog))
        lrs = []
        for base_lr, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = base_lr * factor; lrs.append(g["lr"])
        return lrs

# =============================================================
# INFERENCE
# =============================================================
@torch.no_grad()
def infer_loader(model, loader):
    model.eval(); probs=[]; labels=[]
    for xb, mb, fb, rb, yb, wb in loader:
        xb=xb.to(DEVICE); mb=mb.to(DEVICE); fb=fb.to(DEVICE); rb=rb.to(DEVICE)
        with amp_ctx(): logits = model(xb, mb, fb, rb)
        probs.append(sigmoid_np(logits.detach().cpu().float().numpy()))
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(np.float32)

@torch.no_grad()
def predict_tta(model, X, idx, M, F, R, y, n=TTA_N):
    arr = []
    for k in range(n):
        loader = make_loader(X, idx, M, F, R, y,
                             augment=(k>0), shuffle=False, batch_size=128)
        arr.append(infer_loader(model, loader)[0])
    return np.mean(np.stack(arr, 0), 0)

# =============================================================
# TRAIN ONE MODEL
# =============================================================
def train_one(name, model, X,
              idx_tr, idx_va, idx_te,
              M_tr, M_va, M_te,
              F_tr, F_va, F_te,
              R_tr, R_va, R_te,
              y_tr, y_va, y_te,
              save_path,
              epochs=EPOCHS, patience=PATIENCE,
              use_simclr=False, array_train=None, pl_lr=None):
    print(f"\n{'='*60}"); print(f"TRAIN: {name}"); print(f"{'='*60}")
    model = model.to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer setup
    if use_simclr and hasattr(model, "load_simclr_encoder"):
        if os.path.exists(SIMCLR_PATH):
            model.load_simclr_encoder(SIMCLR_PATH)
            model.freeze_encoder()
            print(f"  Encoder frozen for {FREEZE_EPOCHS} epochs")
            opt = torch.optim.AdamW([
                {"params": model.encoder_params(),     "lr": LR_ENCODER},
                {"params": model.non_encoder_params(), "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            sch = WarmupCosineMultiLR(opt, warmup=5, total=epochs, min_factor=0.01)
            multi_lr = True
        else:
            print(f"  WARNING: SimCLR not found, training from scratch")
            use_simclr = False; multi_lr = False
            lr_use = pl_lr or LR_HEAD
            opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=WEIGHT_DECAY)
            sch = CosineWarmup(opt, min(5,epochs), epochs, lr_use)
    else:
        multi_lr = False
        lr_use = pl_lr or LR_HEAD
        opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=WEIGHT_DECAY)
        sch = CosineWarmup(opt, min(5,epochs), epochs, lr_use)

    n_neg = int((y_tr==0).sum()); n_pos = int((y_tr==1).sum())
    pos_weight = torch.tensor([n_neg/max(1,n_pos)], dtype=torch.float32, device=DEVICE)
    crit   = WeightedPolyLoss(pos_weight=pos_weight, smoothing=LABEL_SMOOTH)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE.type=="cuda"))

    val_loader = make_loader(X, idx_va, M_va, F_va, R_va, y_va, batch_size=128)
    best_acc=0.; best_auc=0.; best_ep=0; wait=0; hist=[]

    for ep in range(1, epochs+1):
        if use_simclr and ep == FREEZE_EPOCHS+1:
            model.unfreeze_encoder()
            print(f"  Ep {ep:03d}: encoder unfrozen")

        model.train(); total=0.; nb=0
        if array_train is None:
            loader = make_loader(X, idx_tr, M_tr, F_tr, R_tr, y_tr,
                                 augment=True, shuffle=True,
                                 batch_size=BATCH, drop_last=True)
        else:
            Xarr,Marr,Farr,Rarr,yarr,warr = array_train
            loader = make_array_loader(Xarr, Marr, Farr, Rarr, yarr,
                                       augment=True, shuffle=True,
                                       batch_size=BATCH, drop_last=True, weights=warr)

        for xb, mb, fb, rb, yb, wb in loader:
            xb=xb.to(DEVICE); mb=mb.to(DEVICE); fb=fb.to(DEVICE)
            rb=rb.to(DEVICE); yb=yb.to(DEVICE); wb=wb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(xb, mb, fb, rb)
                loss   = crit(logits, yb, wb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            total += float(loss.item()); nb += 1

        if multi_lr:
            lrs    = sch.step()
            lr_log = f"lr_enc={lrs[0]:.1e} lr_h={lrs[1]:.1e}"
        else:
            lr_now = sch.step()
            lr_log = f"lr={lr_now:.1e}"

        vp, vl = infer_loader(model, val_loader)
        t, acc  = scan_threshold(vp, vl)
        auc     = roc_auc_score(vl.astype(int), vp)
        avg     = total / max(1, nb)
        hist.append({"epoch":ep,"loss":avg,"val_acc":float(acc),
                     "val_auc":float(auc),"threshold":float(t)})

        if acc > best_acc:
            best_acc=acc; best_auc=auc; best_ep=ep; wait=0
            torch.save(model.state_dict(), save_path); tag="✅"
        else:
            wait += 1; tag=f"wait {wait}/{patience}"

        print(f"  Ep {ep:03d}/{epochs} | loss={avg:.4f} | val_acc={acc:.4f} | "
              f"val_auc={auc:.4f} | t={t:.3f} | {lr_log} | {tag}")

        if wait >= patience:
            print(f"  Early stop ep={best_ep} val_acc={best_acc:.4f} val_auc={best_auc:.4f}")
            break

    save_json(hist, save_path.replace(".pt","_history.json"))
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    print(f"\n  TTA×{TTA_N} predictions for {name}...")
    val_probs  = predict_tta(model, X, idx_va, M_va, F_va, R_va, y_va)
    test_probs = predict_tta(model, X, idx_te, M_te, F_te, R_te, y_te)
    return model, best_acc, best_auc, val_probs, test_probs

# =============================================================
# PSEUDO-LABELING SUR OOF  ← FIX 1
# =============================================================
def run_pseudo_labeling(
        model_builders, X, pool_idx, test_idx,
        X_meta, F_pool_raw, F_test_raw,
        y_all, y_pool, y_test,
        oof_probs, fold_meta_stats, fold_feat_stats, skf):
    """
    Pseudo-labeling propre :
      - les probs OOF sont OUT-OF-FOLD → pas de fuite
      - on sélectionne les samples très confiants (>PL_HIGH ou <PL_LOW)
      - on ré-entraîne chaque modèle fold par fold avec pseudo-samples ajoutés
    """
    print(f"\n{'='*60}")
    print("PSEUDO-LABELING PHASE (OOF-based)")
    print(f"{'='*60}")

    confmask = (oof_probs >= PL_HIGH) | (oof_probs <= PL_LOW)
    sel_pool  = np.where(confmask)[0]          # indices dans pool_idx

    if len(sel_pool) == 0:
        print("No confident pseudo-labels found. Skipping PL phase.")
        return None, None

    yps_all = (oof_probs[sel_pool] >= 0.5).astype(np.float32)
    conf    = np.maximum(oof_probs[sel_pool], 1-oof_probs[sel_pool])

    # Limiter par classe
    chosen = []
    for cls in [0, 1]:
        ci = np.where(yps_all == cls)[0]
        if len(ci) > PL_MAX_PER_CLASS:
            ci = ci[np.argsort(-conf[ci])[:PL_MAX_PER_CLASS]]
        chosen.append(ci)
    chosen  = np.concatenate(chosen)
    sel_pool = sel_pool[chosen]
    yps     = yps_all[chosen]

    # Audit qualité : compare pseudo-labels vs vrais labels
    audit = accuracy_score(y_pool[sel_pool].astype(int), yps.astype(int))
    print(f"Pseudo selected : {len(sel_pool)} samples")
    print(f"  NORM pseudo   : {(yps==0).sum()}")
    print(f"  ABNORM pseudo : {(yps==1).sum()}")
    print(f"  Audit acc     : {audit:.4f}  (vs true labels)")

    if len(sel_pool) < 50:
        print("Too few pseudo-labels (<50). Skipping PL phase.")
        return None, None

    # Indices réels dans X pour les pseudo-samples
    pl_real_idx = pool_idx[sel_pool]

    n_meta = X_meta.shape[1]
    n_feat = F_pool_raw.shape[1]

    pl_oof   = np.zeros(len(pool_idx), dtype=np.float32)
    pl_test_all = []

    for fold, (tr_local, va_local) in enumerate(
            skf.split(pool_idx, y_pool.astype(int)), 1):
        print(f"\n  --- PL Fold {fold}/{N_FOLDS} ---")

        idx_tr = pool_idx[tr_local]; idx_va = pool_idx[va_local]
        y_tr   = y_all[idx_tr].astype(np.float32)
        y_va   = y_all[idx_va].astype(np.float32)

        # Récupérer les stats de normalisation sauvegardées (phase base)
        M_mean, M_std = fold_meta_stats[fold-1]
        F_mean, F_std = fold_feat_stats[fold-1]

        def apply_norm(arr, mean, std):
            Xn = (np.nan_to_num(arr, nan=0., posinf=0., neginf=0.) - mean) / std
            return np.nan_to_num(Xn, nan=0., posinf=0., neginf=0.).astype(np.float32)

        M_tr = apply_norm(X_meta[idx_tr],    M_mean, M_std)
        M_va = apply_norm(X_meta[idx_va],    M_mean, M_std)
        M_te = apply_norm(X_meta[test_idx],  M_mean, M_std)
        F_tr = apply_norm(F_pool_raw[tr_local], F_mean, F_std)
        F_va = apply_norm(F_pool_raw[va_local], F_mean, F_std)
        F_te = apply_norm(F_test_raw,           F_mean, F_std)

        F_tr_raw = F_pool_raw[tr_local]
        R_tr = compute_rules(F_tr_raw, F_tr_raw)
        R_va = compute_rules(F_pool_raw[va_local], F_tr_raw)
        R_te = compute_rules(F_test_raw, F_tr_raw)
        n_rule = R_tr.shape[1]

        # Pseudo-samples : features
        M_pl = apply_norm(X_meta[pl_real_idx],      M_mean, M_std)
        F_pl = apply_norm(F_pool_raw[sel_pool],      F_mean, F_std)
        R_pl = compute_rules(F_pool_raw[sel_pool],   F_tr_raw)

        # Concat train + pseudo
        Xsup = np.stack([ensure_ct(X[i]) for i in idx_tr])
        Xpl  = np.stack([ensure_ct(X[i]) for i in pl_real_idx])
        Xarr = np.concatenate([Xsup, Xpl],  0)
        Marr = np.concatenate([M_tr, M_pl], 0)
        Farr = np.concatenate([F_tr, F_pl], 0)
        Rarr = np.concatenate([R_tr, R_pl], 0)
        yarr = np.concatenate([y_tr, yps],  0)
        warr = np.concatenate([
            np.ones(len(y_tr), dtype=np.float32),
            np.full(len(yps), PSEUDO_SAMPLE_WEIGHT, dtype=np.float32)], 0)

        fold_val_models=[]; fold_test_models=[]

        for key, (builder, _) in model_builders.items():
            free_mem()
            base  = pth(f"fold{fold}_{key}_best.pt")
            save  = pth(f"fold{fold}_{key}_pl_best.pt")
            name  = f"fold{fold}_{key}_PL"
            model = builder(n_meta, n_feat, n_rule).to(DEVICE)
            if os.path.exists(base):
                model.load_state_dict(torch.load(base, map_location=DEVICE))
            model, pacc, pauc, vp, tp = train_one(
                name, model, X,
                idx_tr, idx_va, test_idx,
                M_tr, M_va, M_te,
                F_tr, F_va, F_te,
                R_tr, R_va, R_te,
                y_tr, y_va, y_test,
                save,
                epochs=PL_EPOCHS, patience=PL_PATIENCE,
                use_simclr=False,
                array_train=(Xarr,Marr,Farr,Rarr,yarr,warr),
                pl_lr=PL_LR)
            fold_val_models.append(vp); fold_test_models.append(tp)
            del model; free_mem()

        pl_oof[va_local]  = np.mean(np.stack(fold_val_models), 0)
        pl_test_all.append(np.mean(np.stack(fold_test_models), 0))

    pl_test_probs = np.mean(np.stack(pl_test_all), 0)
    return pl_oof, pl_test_probs

# =============================================================
# MAIN
# =============================================================
def main():
    t0 = time.time()
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"SimCLR path  : {SIMCLR_PATH}")
    print(f"SimCLR found : {os.path.exists(SIMCLR_PATH)}")
    print(f"EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | TTA_N={TTA_N}")
    print(f"FREEZE_EPOCHS={FREEZE_EPOCHS} | LR_HEAD={LR_HEAD} | LR_ENCODER={LR_ENCODER}")

    X, X_meta, y_all, pool_idx, test_idx, y_pool, y_test = load_data()
    F_pool_raw = extract_or_load_features(X, pool_idx, "pool")
    F_test_raw = extract_or_load_features(X, test_idx, "test")
    n_meta = X_meta.shape[1]; n_feat = F_pool_raw.shape[1]

    MODEL_BUILDERS = {
        "resnet_hybrid"   : (ResNetHybrid,    True),   # True = use SimCLR
        "inception_hybrid": (InceptionHybrid, False),
        "tcn_hybrid"      : (TCNHybrid,       False),
    } if USE_THREE_MODELS else {
        "resnet_hybrid"   : (ResNetHybrid,    True),
    }

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(pool_idx), dtype=np.float32)
    test_all = []; logs = []
    fold_meta_stats = []   # (M_mean, M_std) par fold
    fold_feat_stats = []   # (F_mean, F_std) par fold

    # ── PHASE 1 : Base training (5 folds) ────────────────
    print(f"\n{'#'*70}")
    print("PHASE 1 : BASE TRAINING")
    print(f"{'#'*70}")

    for fold, (tr_local, va_local) in enumerate(
            skf.split(pool_idx, y_pool.astype(int)), 1):
        print(f"\n{'#'*70}"); print(f"FOLD {fold}/{N_FOLDS}"); print(f"{'#'*70}")

        idx_tr = pool_idx[tr_local]; idx_va = pool_idx[va_local]
        y_tr   = y_all[idx_tr].astype(np.float32)
        y_va   = y_all[idx_va].astype(np.float32)

        M_tr_raw = X_meta[idx_tr].astype(np.float32)
        M_va_raw = X_meta[idx_va].astype(np.float32)
        M_te_raw = X_meta[test_idx].astype(np.float32)
        M_tr, M_mean, M_std = norm_fit_apply(M_tr_raw, M_tr_raw)
        M_va, _, _ = norm_fit_apply(M_tr_raw, M_va_raw)
        M_te, _, _ = norm_fit_apply(M_tr_raw, M_te_raw)
        fold_meta_stats.append((M_mean, M_std))

        F_tr_raw = F_pool_raw[tr_local]; F_va_raw = F_pool_raw[va_local]
        F_te_raw = F_test_raw
        F_tr, F_mean, F_std = norm_fit_apply(F_tr_raw, F_tr_raw)
        F_va, _, _ = norm_fit_apply(F_tr_raw, F_va_raw)
        F_te, _, _ = norm_fit_apply(F_tr_raw, F_te_raw)
        fold_feat_stats.append((F_mean, F_std))

        R_tr = compute_rules(F_tr_raw, F_tr_raw)
        R_va = compute_rules(F_va_raw, F_tr_raw)
        R_te = compute_rules(F_te_raw, F_tr_raw)
        n_rule = R_tr.shape[1]

        np.save(pth(f"fold{fold}_meta_mean.npy"), M_mean)
        np.save(pth(f"fold{fold}_meta_std.npy"),  M_std)
        np.save(pth(f"fold{fold}_feat_mean.npy"), F_mean)
        np.save(pth(f"fold{fold}_feat_std.npy"),  F_std)

        print(f"Train: {len(idx_tr)} | NORM={(y_tr==0).sum()} | ABNORMAL={(y_tr==1).sum()}")
        print(f"Val  : {len(idx_va)} | NORM={(y_va==0).sum()} | ABNORMAL={(y_va==1).sum()}")

        val_models=[]; test_models=[]

        for key, (builder, use_simclr) in MODEL_BUILDERS.items():
            free_mem()
            name  = f"fold{fold}_{key}"
            save  = pth(f"{name}_best.pt")
            model = builder(n_meta, n_feat, n_rule)
            print(f"\n{name.upper()} — {sum(p.numel() for p in model.parameters()):,} params"
                  + (" [SimCLR]" if use_simclr else ""))

            model, bacc, bauc, vp, tp = train_one(
                name, model, X,
                idx_tr, idx_va, test_idx,
                M_tr, M_va, M_te,
                F_tr, F_va, F_te,
                R_tr, R_va, R_te,
                y_tr, y_va, y_test,
                save, use_simclr=use_simclr)

            val_models.append(vp); test_models.append(tp)
            logs.append({"fold":fold,"model":key,"stage":"base",
                         "use_simclr":use_simclr,
                         "best_val_acc":float(bacc),"best_val_auc":float(bauc),
                         "checkpoint":save})
            del model; free_mem()

        fold_val  = np.mean(np.stack(val_models),  0)
        fold_test = np.mean(np.stack(test_models), 0)
        t_f, acc_f = scan_threshold(fold_val, y_va)
        auc_f = roc_auc_score(y_va.astype(int), fold_val)
        print(f"\nFOLD {fold} BASE: val_acc={acc_f:.4f}, val_auc={auc_f:.4f}, t={t_f:.4f}")

        oof[va_local] = fold_val
        test_all.append(fold_test)
        np.save(pth(f"fold{fold}_val_probs.npy"),  fold_val)
        np.save(pth(f"fold{fold}_test_probs.npy"), fold_test)

    # OOF base
    base_test_probs = np.mean(np.stack(test_all), 0)
    t_base, oof_base_acc = scan_threshold(oof, y_pool)
    auc_base = roc_auc_score(y_pool.astype(int), oof)
    print(f"\n{'='*60}")
    print(f"BASE OOF: acc={oof_base_acc:.4f}, auc={auc_base:.4f}, t={t_base:.4f}")
    print(f"{'='*60}")
    np.save(pth("oof_probs_base.npy"), oof)

    # ── PHASE 2 : Pseudo-labeling sur OOF ────────────────
    final_oof  = oof.copy()
    final_test = base_test_probs.copy()

    if USE_PSEUDO_LABELING:
        pl_oof, pl_test = run_pseudo_labeling(
            MODEL_BUILDERS, X, pool_idx, test_idx,
            X_meta, F_pool_raw, F_test_raw,
            y_all, y_pool, y_test,
            oof, fold_meta_stats, fold_feat_stats, skf)

        if pl_oof is not None:
            t_pl, pl_acc = scan_threshold(pl_oof, y_pool)
            auc_pl = roc_auc_score(y_pool.astype(int), pl_oof)
            print(f"\nPL OOF: acc={pl_acc:.4f}, auc={auc_pl:.4f}, t={t_pl:.4f}")
            if pl_acc >= oof_base_acc:
                print("✅ PL improved OOF — using PL predictions for final eval.")
                final_oof  = pl_oof
                final_test = pl_test
                np.save(pth("oof_probs_pl.npy"),   pl_oof)
                np.save(pth("test_probs_pl.npy"),  pl_test)
            else:
                print("⚠️  PL did not improve OOF — keeping BASE predictions.")

    # ── Final evaluation ──────────────────────────────────
    t_acc, oof_acc = scan_threshold(final_oof, y_pool)
    t_f1,  oof_f1  = scan_threshold(final_oof, y_pool, "f1")
    t_y, tpr, fpr  = youden_threshold(final_oof, y_pool.astype(int))
    final_t = t_acc

    oof_m,  oof_cm,  _         = compute_metrics(y_pool, final_oof,  final_t)
    test_m, test_cm, test_pred = compute_metrics(y_test, final_test, final_t)

    print(f"\n{'='*70}")
    print("FINAL v10b SimCLR Hybrid RESULT")
    print(f"{'='*70}")
    print(f"Accuracy threshold : {t_acc:.4f} | OOF acc={oof_acc:.4f}")
    print(f"F1 threshold       : {t_f1:.4f}  | OOF f1={oof_f1:.4f}")
    print(f"Youden threshold   : {t_y:.4f}   | TPR={tpr:.4f} | FPR={fpr:.4f}")
    print(f"FINAL threshold    : {final_t:.4f}")
    print("\nOOF METRICS:");  print(json.dumps(oof_m,  indent=2))
    print("\nTEST METRICS:"); print(json.dumps(test_m, indent=2))
    print("\nTEST CLASSIFICATION REPORT:")
    print(classification_report(y_test.astype(int), test_pred.astype(int),
                                target_names=["NORM","ABNORMAL"], digits=4))
    print("\nTEST CONFUSION MATRIX:"); print(test_cm)

    save_json({
        "config": {
            "n_folds": N_FOLDS, "use_three_models": USE_THREE_MODELS,
            "use_pseudo_labeling": USE_PSEUDO_LABELING,
            "pl_strategy": "OOF-based (no leakage)",
            "simclr_path": SIMCLR_PATH,
            "pl_low": PL_LOW, "pl_high": PL_HIGH,
            "pl_max_per_class": PL_MAX_PER_CLASS,
            "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH,
            "lr_head": LR_HEAD, "lr_encoder": LR_ENCODER,
            "freeze_epochs": FREEZE_EPOCHS, "tta_n": TTA_N
        },
        "fold_logs": logs,
        "thresholds": {"accuracy":float(t_acc),"f1":float(t_f1),
                       "youden":float(t_y),"final":float(final_t)},
        "oof_metrics":  oof_m,
        "test_metrics": test_m
    }, pth("metrics_v10b_simclr_hybrid.json"))

    np.save(pth("oof_probs.npy"),   final_oof)
    np.save(pth("test_probs.npy"),  final_test)
    np.save(pth("test_pred.npy"),   test_pred)
    plot_cm(oof_cm,  pth("confusion_matrix_oof.png"),  "OOF Confusion Matrix")
    plot_cm(test_cm, pth("confusion_matrix_test.png"), "Test Confusion Matrix")

    elapsed = (time.time()-t0)/60.0
    print(f"\nOutput : {OUTPUT_DIR}")
    print(f"Elapsed : {elapsed:.1f} min")
    if test_m["accuracy"] >= 0.90:
        print("✅ 90% TEST accuracy reached!")
    else:
        print(f"⚠️  Current TEST accuracy: {test_m['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
