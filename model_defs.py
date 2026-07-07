import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEADS = 12
N_SAMPLES = 5000
FS = 500


def ensure_ct(sig):
    sig = np.asarray(sig, dtype=np.float32)
    if sig.shape == (12, 5000):
        return sig
    if sig.shape == (5000, 12):
        return sig.T
    raise ValueError(f"Bad ECG shape: {sig.shape}")


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, hid),
            nn.GELU(),
            nn.Linear(hid, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class MultiPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            [x.mean(-1), x.max(-1).values, x.std(-1)],
            dim=1
        )


class ResBlock(nn.Module):
    def __init__(self, ic, oc, k=7, stride=1, dil=1, drop=0.1):
        super().__init__()
        pad = dil * (k - 1) // 2

        self.c1 = nn.Conv1d(ic, oc, k, stride=stride, padding=pad, dilation=dil, bias=False)
        self.b1 = nn.BatchNorm1d(oc)

        self.c2 = nn.Conv1d(oc, oc, k, padding=pad, dilation=dil, bias=False)
        self.b2 = nn.BatchNorm1d(oc)

        self.se = SEBlock(oc)
        self.dp = nn.Dropout(drop)

        self.sk = nn.Sequential(
            nn.Conv1d(ic, oc, 1, stride=stride, bias=False),
            nn.BatchNorm1d(oc)
        ) if (ic != oc or stride != 1) else nn.Identity()

    def forward(self, x):
        s = self.sk(x)
        x = F.gelu(self.b1(self.c1(x)))
        x = self.dp(x)
        x = self.se(self.b2(self.c2(x)))
        return F.gelu(x + s)


class MetaBranch(nn.Module):
    def __init__(self, n, out=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class FeatureBranch(MetaBranch):
    pass


class RuleBranch(nn.Module):
    def __init__(self, n, out=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, out),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class ResNetHybrid(nn.Module):
    def __init__(self, n_meta=7, n_feat=16, n_rule=10, n_outputs=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1)
        )

        self.layers = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2, drop=0.12),
            ResBlock(128, 128, dil=2, drop=0.12),
            ResBlock(128, 256, stride=2, drop=0.15),
            ResBlock(256, 256, dil=4, drop=0.15),
            ResBlock(256, 512, stride=2, drop=0.18),
            ResBlock(512, 512, dil=8, drop=0.18)
        )

        self.pool = MultiPool()

        self.meta = MetaBranch(n_meta)
        self.feat = FeatureBranch(n_feat)
        self.rule = RuleBranch(n_rule)

        self.head = nn.Sequential(
            nn.Linear(512 * 3 + 32 + 32 + 16, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, n_outputs)
        )

    def forward(self, x, m, f, r):
        z = torch.cat([
            self.pool(self.layers(self.stem(x))),
            self.meta(m),
            self.feat(f),
            self.rule(r)
        ], dim=1)

        out = self.head(z)
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out


def rpeak_proxy(lead):
    x = lead.astype(np.float32) - np.median(lead)
    d = np.abs(np.diff(x, prepend=x[0]))
    e = np.convolve(d, np.ones(9, dtype=np.float32) / 9.0, mode="same")

    thr = e.mean() + 1.2 * e.std()
    min_dist = int(0.25 * FS)

    peaks = []
    last = -min_dist

    for i in range(1, len(e) - 1):
        if i - last >= min_dist and e[i] > thr and e[i] >= e[i - 1] and e[i] >= e[i + 1]:
            peaks.append(i)
            last = i

    return np.array(peaks, dtype=np.int32), e


def extract_features_one(sig):
    sig = np.nan_to_num(
        ensure_ct(sig).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    lead = sig[1]
    peaks, der = rpeak_proxy(lead)

    n = len(peaks)
    hr = 60.0 * n / (N_SAMPLES / FS)

    if n >= 3:
        rr = np.diff(peaks) / FS
        rr_mean = float(rr.mean())
        rr_std = float(rr.std())
        rr_cv = rr_std / max(1e-6, rr_mean)
    else:
        rr_mean = rr_std = rr_cv = 0.0

    widths = []
    for p in peaks[:30]:
        l = max(0, p - int(0.12 * FS))
        r = min(N_SAMPLES, p + int(0.12 * FS))
        seg = der[l:r]

        if len(seg) > 3:
            active = np.where(seg > seg.mean() + 0.5 * seg.std())[0]
            if len(active) > 1:
                widths.append((active[-1] - active[0]) / FS)

    qrs = float(np.median(widths)) if widths else 0.0

    abs_sig = np.abs(sig)
    le = np.mean(sig ** 2, axis=1)

    win = int(0.8 * FS)
    if win % 2 == 0:
        win += 1

    baseline = np.convolve(
        lead,
        np.ones(win, dtype=np.float32) / win,
        mode="same"
    )

    feats = np.array([
        hr,
        rr_mean,
        rr_std,
        rr_cv,
        qrs,
        float(np.mean(sig ** 2)),
        float(abs_sig.mean()),
        float(abs_sig.std()),
        float(le.mean()),
        float(le.std()),
        float(le.max()),
        float(le.min()),
        float(np.mean(baseline ** 2)),
        float(np.mean(der)),
        float(np.mean(np.std(sig, axis=1) < 1e-5)),
        float(n)
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def compute_rules(F, ref):
    F = np.asarray(F, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    if F.ndim == 1:
        F = F[None, :]

    e_hi = np.percentile(ref[:, 5], 90)
    e_lo = np.percentile(ref[:, 5], 5)
    n_hi = np.percentile(ref[:, 13], 90)
    b_hi = np.percentile(ref[:, 12], 90)

    hr = F[:, 0]
    rr_cv = F[:, 3]
    qrs = F[:, 4]
    energy = F[:, 5]
    base = F[:, 12]
    noise = F[:, 13]
    flat = F[:, 14]

    rules = [
        ((hr > 0) & (hr < 50)),
        hr > 110,
        rr_cv > 0.18,
        qrs > 0.115,
        energy > e_hi,
        energy < e_lo,
        noise > n_hi,
        base > b_hi,
        flat > 0.15
    ]

    R = np.stack([x.astype(np.float32) for x in rules], axis=1)
    score = np.clip(R.sum(1, keepdims=True) / 9.0, 0, 1)

    return np.concatenate([R, score], axis=1).astype(np.float32)


def apply_norm(x, mean, std):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_resnet_hybrid(path, n_outputs):
    model = ResNetHybrid(n_meta=7, n_feat=16, n_rule=10, n_outputs=n_outputs)
    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model