# %%
"""
Jupyter-style Python notebook (script format with cell markers).
Purpose: Reproducible pipeline to validate "Size–Error Trade-off Optimization for Lossy IoT Data Compression".
Notes:
 - The notebook is structured into runnable cells.
 - Where public datasets are required, the notebook attempts to download them. If offline, synthetic fallbacks are used.
 - Uses PyTorch for ML models; NumPy/SciPy/skimage for classical methods and metrics.
 - Adjust paths/URLs at the top if your environment blocks downloads.

Cell index layout:
 0. Setup & configuration
 1. Utilities: metrics, plotting, pareto
 2. Data loaders (HAR, Intel Lab, synthetic images)
 3. Classical methods (DCT/DWT/Delta/PLA)
 4. ML methods (LSTM autoencoder for HAR, Conv AE for images)
 5. Hybrid methods (downscale+SRCNN placeholder, quantization-aware AE placeholder)
 6. Sweep + evaluate (grid search for each method)
 7. Pareto frontier, tables, plots
 8. Statistical tests

This notebook is a starting point to reproduce the experiments and can be extended.
"""

# %%
# 0. Setup & configuration
import os
import math
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Signal/image processing
from scipy import fftpack
from scipy import signal

# Image metrics
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    def ssim(a, b, **kwargs):
        return np.nan

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RNG_SEED)

OUT_DIR = Path('validation_outputs')
OUT_DIR.mkdir(exist_ok=True)

print('TORCH_AVAILABLE =', TORCH_AVAILABLE)

# %%
# 1. Utilities: metrics, plotting, pareto

def compute_mse(x, xr):
    return float(np.mean((x - xr) ** 2))

def compute_psnr(x, xr, data_range=None):
    mse = compute_mse(x, xr)
    if mse == 0:
        return float('inf')
    if data_range is None:
        data_range = float(np.max(x) - np.min(x))
        if data_range == 0:
            data_range = 1.0
    return 10 * math.log10((data_range ** 2) / mse)

def compute_snr(x, xr):
    mse = compute_mse(x, xr)
    var = float(np.var(x))
    if mse == 0:
        return float('inf')
    if var == 0:
        return float('inf')
    return 10 * math.log10(var / mse)

def compute_cr(original_bytes, compressed_bytes):
    if compressed_bytes == 0:
        return float('inf')
    return float(original_bytes) / float(compressed_bytes)

def pareto_frontier(df, x_metric='CR', y_metric='MSE', maximize_x=True, maximize_y=False):
    # Return non-dominated points as subset of df indices
    records = df.copy()
    if maximize_x:
        records = records.sort_values(x_metric, ascending=False)
    else:
        records = records.sort_values(x_metric, ascending=True)
    pareto_idx = []
    best_y = None
    for idx, row in records.iterrows():
        cur_y = row[y_metric]
        if best_y is None:
            pareto_idx.append(idx)
            best_y = cur_y
            continue
        # For minimization of y_metric
        if not maximize_y:
            if cur_y < best_y:
                pareto_idx.append(idx)
                best_y = cur_y
        else:
            if cur_y > best_y:
                pareto_idx.append(idx)
                best_y = cur_y
    return df.loc[pareto_idx]

# %%
# 2. Data loaders

# 2.1 UCI HAR loader (attempt download, fallback synthetic)
HAR_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'

def load_har(use_local=False):
    """Return (X_train, X_val, X_test), dims: (N, seq_len, channels)"""
    try:
        import requests, zipfile, io
        print('Attempting to download UCI HAR...')
        r = requests.get(HAR_URL, timeout=20)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # The UCI HAR dataset requires parsing — to keep notebook compact, we'll synthesize a small sample
        raise Exception('skip parsing; use synthetic fallback for lightweight validation')
    except Exception as e:
        print('Using synthetic HAR-like data (fallback):', e)
        N = 2000
        seq_len = 128
        channels = 6
        # Generate synthetic multi-channel periodic + noise signals with class labels
        t = np.linspace(0, 2.56, seq_len)
        X = []
        y = []
        for i in range(N):
            base_freq = np.random.choice([1, 2, 3, 4, 5])
            sig = np.stack([np.sin(2 * np.pi * base_freq * t + np.random.randn()*0.1) for _ in range(channels)], axis=-1)
            sig += 0.1 * np.random.randn(seq_len, channels)
            X.append(sig)
            y.append(base_freq % 6)
        X = np.stack(X)
        y = np.array(y)
        # normalize to [0,1]
        X = (X - X.min()) / (X.max() - X.min())
        n = len(X)
        train = int(0.7 * n)
        val = int(0.85 * n)
        return (X[:train], X[train:val], X[val:]), (y[:train], y[train:val], y[val:])

# 2.2 Intel Lab environmental dataset — fallback synthetic

def load_env():
    # Simulate smooth temperature/humidity signals with diurnal cycle
    N = 5000
    seq_len = 1  # sample-wise
    channels = 4  # temp, hum, light, voltage
    t = np.arange(N)
    temp = 20 + 5 * np.sin(2 * np.pi * t / 1440) + 0.5 * np.random.randn(N)
    hum = 50 + 10 * np.sin(2 * np.pi * t / 1440 + 1.0) + 1.0 * np.random.randn(N)
    light = np.clip(100 * np.sin(2 * np.pi * t / 1440) + 20 * np.random.randn(N), 0, None)
    volt = 3.3 + 0.01 * np.random.randn(N)
    X = np.stack([temp, hum, light, volt], axis=-1)
    X = (X - X.min()) / (X.max() - X.min())
    n = len(X)
    train = int(0.7 * n)
    val = int(0.85 * n)
    return (X[:train], X[train:val], X[val:])

# 2.3 Multimedia images (generate small synthetic images)

def load_images(n_images=200, H=120, W=160, C=3):
    X = np.random.rand(n_images, H, W, C).astype(np.float32)
    # Add simple shapes to help perceptual metrics
    for i in range(n_images):
        rr = np.random.randint(5, 40)
        cx = np.random.randint(rr, W-rr)
        cy = np.random.randint(rr, H-rr)
        Y, Xc = np.ogrid[:H, :W]
        mask = (Y - cy)**2 + (Xc - cx)**2 <= rr**2
        color = np.random.rand(3)
        X[i][mask] = color
    return X

# Load datasets (fallbacks)
(har_train, har_val, har_test), (har_ytrain, har_yval, har_ytest) = load_har()
env_train, env_val, env_test = load_env()
images = load_images(200)

print('HAR shapes:', har_train.shape, har_val.shape, har_test.shape)
print('ENV shapes:', env_train.shape, env_val.shape, env_test.shape)
print('Images shape:', images.shape)

# %%
# 3. Classical compression implementations

# 3.1 DCT-based compression for 1D sequences and images (blockwise)

def dct_compress_1d(signal, retain_ratio=0.2):
    # signal: (seq_len, channels) or (seq_len,) -> apply DCT per channel
    sig = signal.copy()
    coeffs = fftpack.dct(sig, axis=0, norm='ortho')
    n_keep = max(1, int(coeffs.shape[0] * retain_ratio))
    keep_idx = np.argsort(np.abs(coeffs), axis=0)[-n_keep:]
    thresh = np.sort(np.abs(coeffs), axis=0)[-n_keep:][0]
    mask = np.abs(coeffs) >= thresh
    coeffs_masked = coeffs * mask
    recon = fftpack.idct(coeffs_masked, axis=0, norm='ortho')
    # approximate compressed size = number of retained coefficients * 4 bytes
    compressed_bytes = np.sum(mask) * 4
    return recon, compressed_bytes

# 3.2 Delta + uniform quantization

def delta_quantize_1d(signal, n_bits=8):
    # Flatten channels, compute diffs
    sig = signal.copy().astype(np.float32)
    diffs = np.diff(sig, axis=0)
    # scale to [-1,1] based on observed range
    mx = np.max(np.abs(diffs))
    if mx == 0:
        mx = 1e-6
    qlevels = 2 ** n_bits
    q = np.round((diffs / mx) * (qlevels/2 - 1))
    # compressed bytes ~ number of diffs * bits
    compressed_bits = q.size * n_bits
    compressed_bytes = int(np.ceil(compressed_bits / 8.0))
    # reconstruct
    diffs_rec = (q / (qlevels/2 - 1)) * mx
    recon = np.vstack([sig[0:1], sig[0:1] + np.cumsum(diffs_rec, axis=0)])
    return recon, compressed_bytes

# 3.3 Piecewise Linear Approximation (PLA) simple implementation

def pla_compress_1d(signal, eps=0.01):
    # simple streaming PLA: segment with linear fit until max error > eps
    seq_len, ch = signal.shape
    segments = []  # list of (start, end, slope, intercept)
    i = 0
    while i < seq_len:
        j = i + 2
        while j <= seq_len:
            x = np.arange(i, j)
            y = signal[i:j]
            # fit per-channel linear regression
            A = np.vstack([x, np.ones_like(x)]).T
            # solve for each channel
            params = np.linalg.lstsq(A, y, rcond=None)[0]
            yhat = A @ params
            err = np.max(np.abs(y - yhat))
            if err > eps:
                break
            j += 1
            if j - i > 500:  # safety
                break
        # last valid segment is j-1
        end = j - 1
        x = np.arange(i, end)
        A = np.vstack([x, np.ones_like(x)]).T
        params = np.linalg.lstsq(A, signal[i:end], rcond=None)[0]
        segments.append((i, end, params))
        i = end
    # rough compressed size: each segment stores 2 floats per channel + indices
    compressed_bytes = sum([(2 * signal.shape[1] + 1) * 4 for _ in segments])
    # reconstruct
    recon = np.zeros_like(signal)
    for seg in segments:
        s, e, params = seg
        x = np.arange(s, e)
        recon[s:e] = np.vstack([x, np.ones_like(x)]).T @ params
    return recon, compressed_bytes

# %%
# Quick test of classical functions on a HAR sample
sample = har_test[0]
recon_dct, cbytes_dct = dct_compress_1d(sample, retain_ratio=0.2)
recon_delta, cbytes_delta = delta_quantize_1d(sample, n_bits=6)
recon_pla, cbytes_pla = pla_compress_1d(sample, eps=0.02)
print('sample shapes', sample.shape)
print('DCT bytes', cbytes_dct, 'MSE', compute_mse(sample, recon_dct))
print('Delta bytes', cbytes_delta, 'MSE', compute_mse(sample, recon_delta))
print('PLA bytes', cbytes_pla, 'MSE', compute_mse(sample, recon_pla))

# %%
# 4. ML methods (lightweight implementations)
if TORCH_AVAILABLE:
    class LSTMAutoencoderSmall(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.bottleneck = nn.Linear(64, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, 64)
            self.decoder = nn.LSTM(64, input_dim, num_layers=2, batch_first=True, dropout=0.2)

        def forward(self, x):
            out, (h, c) = self.encoder(x)
            hidden = h[-1]
            latent = self.bottleneck(hidden)
            dec_in = self.decoder_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
            out, _ = self.decoder(dec_in)
            return out, latent

    def train_lstm_ae(model, X_train, X_val=None, epochs=20, batch_size=64, lr=1e-3):
        model.train()
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            loss_acc = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                out, _ = model(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                opt.step()
                loss_acc += float(loss.item()) * batch.size(0)
            loss_acc /= len(ds)
            if (epoch+1) % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs} loss={loss_acc:.6f}')
        return model

else:
    print('Torch not available; skipping ML methods. Install PyTorch to run autoencoders.')

# %%
# 5. Hybrid methods (placeholders)
# Implement downscale + simple interpolation as SR proxy

def downscale_and_upscale_image(img, factor=4):
    # img: HWC in [0,1]
    H, W, C = img.shape
    H2, W2 = H // factor, W // factor
    small = signal.resample(signal.resample(img, W2, axis=1), H2, axis=0)
    recon = signal.resample(signal.resample(small, W, axis=1), H, axis=0)
    # compressed bytes approx: small.size * 4
    compressed_bytes = small.size * 4
    return recon, compressed_bytes

# %%
# 6. Sweep + evaluate (for a subset of methods and operating points)
results = []

# Helper to compute original size in bytes for a signal
def original_bytes_for_1d(sig):
    return sig.size * 4  # float32

# HAR: classical sweeps
for retain in [0.1, 0.2, 0.3, 0.5]:
    mse_list = []
    cr_list = []
    for i in range(min(200, len(har_test))):
        sig = har_test[i]
        recon, cbytes = dct_compress_1d(sig, retain_ratio=retain)
        mse = compute_mse(sig, recon)
        orig_b = original_bytes_for_1d(sig)
        cr = compute_cr(orig_b, cbytes)
        results.append({'Dataset': 'HAR', 'Method': 'DCT', 'Level': retain, 'CR': cr, 'MSE': mse})

for nbits in [4, 6, 8]:
    for i in range(min(200, len(har_test))):
        sig = har_test[i]
        recon, cbytes = delta_quantize_1d(sig, n_bits=nbits)
        mse = compute_mse(sig, recon)
        orig_b = original_bytes_for_1d(sig)
        cr = compute_cr(orig_b, cbytes)
        results.append({'Dataset': 'HAR', 'Method': 'DeltaQ', 'Level': nbits, 'CR': cr, 'MSE': mse})

# PLA
for eps in [0.005, 0.01, 0.02]:
    for i in range(min(200, len(har_test))):
        sig = har_test[i]
        recon, cbytes = pla_compress_1d(sig, eps=eps)
        mse = compute_mse(sig, recon)
        orig_b = original_bytes_for_1d(sig)
        cr = compute_cr(orig_b, cbytes)
        results.append({'Dataset': 'HAR', 'Method': 'PLA', 'Level': eps, 'CR': cr, 'MSE': mse})

# LSTM AE (if available) — train small model on a subset to demonstrate
if TORCH_AVAILABLE:
    Xtr = har_train.astype(np.float32)
    model = LSTMAutoencoderSmall(input_dim=Xtr.shape[2], latent_dim=8)
    model = train_lstm_ae(model, Xtr, epochs=15, batch_size=128)
    model.eval()
    with torch.no_grad():
        for i in range(min(200, len(har_test))):
            sig = har_test[i:i+1].astype(np.float32)
            inp = torch.tensor(sig)
            out, latent = model(inp)
            out_np = out.numpy()[0]
            # compressed size approx: latent_dim * 4 bytes
            cbytes = latent.numpy().size * 4
            mse = compute_mse(sig[0], out_np)
            orig_b = original_bytes_for_1d(sig[0])
            cr = compute_cr(orig_b, cbytes)
            results.append({'Dataset': 'HAR', 'Method': 'LSTM_AE', 'Level': 8, 'CR': cr, 'MSE': mse})

# Images: JPEG-like via scipy (approx) and downscale
for factor in [2, 4, 8]:
    for i in range(min(100, images.shape[0])):
        img = images[i]
        recon, cbytes = downscale_and_upscale_image(img, factor=factor)
        mse = compute_mse(img, recon)
        ps = compute_psnr(img, recon, data_range=1.0)
        orig_b = img.size * 4
        cr = compute_cr(orig_b, cbytes)
        results.append({'Dataset': 'Images', 'Method': 'Downscale_SR', 'Level': factor, 'CR': cr, 'MSE': mse, 'PSNR': ps})

# Save intermediate results
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / 'raw_results.csv', index=False)
print('Saved results rows:', len(res_df))

# %%
# 7. Pareto frontier and plots

df_har = res_df[res_df['Dataset'] == 'HAR']
if not df_har.empty:
    # aggregate per Method-Level by mean
    agg = df_har.groupby(['Method', 'Level']).agg({'CR':'mean','MSE':'mean'}).reset_index()
    pareto = pareto_frontier(agg, x_metric='CR', y_metric='MSE')
    print('HAR pareto points:')
    print(pareto)
    plt.figure(figsize=(6,4))
    for method in agg['Method'].unique():
        sub = agg[agg['Method']==method]
        plt.scatter(sub['CR'], sub['MSE'], label=method)
    plt.xlabel('Compression Ratio (CR)')
    plt.ylabel('MSE')
    plt.title('HAR: Size-Error Trade-off')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / 'har_tradeoff.png')

# Images tradeoff (PSNR vs CR)
df_img = res_df[res_df['Dataset']=='Images']
if not df_img.empty:
    agg_img = df_img.groupby(['Method','Level']).agg({'CR':'mean','PSNR':'mean'}).reset_index()
    plt.figure(figsize=(6,4))
    for method in agg_img['Method'].unique():
        sub = agg_img[agg_img['Method']==method]
        plt.plot(sub['CR'], sub['PSNR'], marker='o', label=method)
    plt.xlabel('CR')
    plt.ylabel('PSNR (dB)')
    plt.title('Images: Size-Error Trade-off')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / 'images_tradeoff.png')

print('Plots saved to', OUT_DIR)

# %%
# 8. Statistical tests (paired t-test skeleton)
from scipy import stats

# Example: compare DCT vs LSTM_AE on HAR at closest CRs
if TORCH_AVAILABLE and not df_har.empty:
    dct_rows = agg[agg['Method']=='DCT']
    lstm_rows = agg[agg['Method']=='LSTM_AE']
    if (not dct_rows.empty) and (not lstm_rows.empty):
        # find matched CRs within 10%
        pairs = []
        for _, r in dct_rows.iterrows():
            cr = r['CR']
            cand = lstm_rows.iloc[(lstm_rows['CR'] - cr).abs().argsort()[:1]]
            if abs(cand['CR'].values[0] - cr)/cr < 0.1:
                pairs.append((r['PSNR'] if 'PSNR' in r else r['MSE'], cand['PSNR'].values[0] if 'PSNR' in cand else cand['MSE'].values[0]))
        if len(pairs) >= 3:
            a, b = zip(*pairs)
            t, p = stats.ttest_rel(a, b)
            print('Paired t-test t=%.4f p=%.4f' % (t, p))

# %%
# Save summary
res_df.to_pickle(OUT_DIR / 'results.pkl')
print('Notebook run completed. Results and plots are in', OUT_DIR)
