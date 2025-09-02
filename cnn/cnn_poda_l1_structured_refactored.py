# cnn_poda_ln_structured_refactored.py
# ============================================================
# CNN 1D — PODA ESTRUTURADA L1 COM RECONSTRUÇÃO DO MODELO
# Remove filtros com menor norma L1 e reconstrói o modelo com menos filtros.
# Sparsities: 20%, 40%, 60%, 80%
# ============================================================

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune

from sklearn.preprocessing import StandardScaler

import threading
import psutil
import board
import busio
from adafruit_ina219 import INA219

# ----------------------- Config -----------------------
CSV_PATH        = "bd_EstacaoVargemFria_e_Pesca.csv"
SEED            = 42
DEVICE          = torch.device("cpu")
EPOCHS_BASELINE = 100
LR_BASELINE     = 1e-3
BATCH_SIZE      = 32
N_SPLITS        = 5
SAMPLE_INTERVAL = 0.01
INA_ADDR        = 0x40
OUT_CSV         = "resultados_cnn_poda_ln_structured_refactored.csv"
SPARSITY_LEVELS = [0.2, 0.4, 0.6, 0.8]

# ----------------------- Utils -----------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_cnn1d_shape(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).reshape(x.shape[0], 1, x.shape[1])

def mae(y_true, y_pred): return float(np.mean(np.abs(y_pred - y_true)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
def r2_score_np(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ----------------------- Dataset -----------------------
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.float32).reshape(-1))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ----------------------- Modelo Base -----------------------
def build_cnn_model(n_filters1=8, n_filters2=16, length=5):
    return nn.Sequential(
        nn.Conv1d(1, n_filters1, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv1d(n_filters1, n_filters2, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(n_filters2 * length, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# ----------------------- Energia -----------------------
class EnergyMeter:
    def __init__(self, sample_interval=0.01, addr=0x40):
        self.sample_interval = sample_interval
        self.addr = addr
        self.samples = deque(); self.enabled = True
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = INA219(self.i2c, addr=self.addr)
            self.sensor.set_calibration_32V_2A()
        except Exception as e:
            print(f"[AVISO] Falha ao inicializar INA219: {e}"); self.enabled = False
    def start(self):
        if not self.enabled: return
        self.samples.clear()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
    def _loop(self):
        self.start_t = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                v = float(self.sensor.bus_voltage); i = float(self.sensor.current)
                p = v * (i / 1000.0); self.samples.append((t,v,i,p))
            except: self.samples.append((t,np.nan,np.nan,np.nan))
            time.sleep(self.sample_interval)
        self.end_t = time.perf_counter()
    def stop(self):
        if not self.enabled: return
        self._stop.set(); self._thread.join(timeout=2.0)
    def summarize(self):
        if not self.enabled or len(self.samples) < 2: return {}
        arr = np.array(self.samples,dtype=float)
        t,V,I,P = arr[:,0],arr[:,1],arr[:,2],arr[:,3]
        mask = np.isfinite(P); t, P = t[mask], P[mask]
        energy = 0.0
        if len(P)>=2: energy = np.sum((P[:-1]+P[1:])/2*np.diff(t))
        return {"duration_s": float(self.end_t-self.start_t),
                "energy_J": energy,
                "avg_power_W": float(np.nanmean(P)),
                "peak_power_W": float(np.nanmax(P)),
                "avg_current_mA": float(np.nanmean(I)),
                "peak_current_mA": float(np.nanmax(I)),
                "avg_voltage_V": float(np.nanmean(V)),
                "n_samples": len(self.samples)}

# ----------------------- Avaliação -----------------------
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    meter = EnergyMeter(sample_interval=SAMPLE_INTERVAL, addr=INA_ADDR)
    with torch.no_grad():
        for xb, _ in loader: _ = model(xb); break

    mem_before = get_memory_usage_mb()
    if meter.enabled: meter.start(); time.sleep(0.05)
    t0 = time.time(); n_preds = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).cpu().numpy()
            y_pred.append(preds); y_true.append(yb.cpu().numpy())
            n_preds += len(yb)
    t1 = time.time()
    if meter.enabled: meter.stop()

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "total_inference_s": t1 - t0,
        "time_per_sample_ms": (t1 - t0) * 1000.0 / n_preds,
        **meter.summarize(),
        "energy_uJ_per_inf": (meter.summarize().get("energy_J", 0) * 1e6 / n_preds) if n_preds else None,
        "mem_mb_before_energy_infer": mem_before
    }

# ----------------------- Walk-forward -----------------------
def walk_forward_split(n, n_splits=5):
    fold_size = n // (n_splits+1)
    return [(range(0, fold_size*(i+1)), range(fold_size*(i+1), fold_size*(i+2))) for i in range(n_splits)]

# ----------------------- Main -----------------------
def main():
    set_seed(SEED)
    df = pd.read_csv(CSV_PATH)
    df = df[df['Nome'].str.strip()=="Estação Pesca - UFRPE"]
    df['Data estação'] = pd.to_datetime(df['Data estação'], errors='coerce')
    df = df.sort_values('Data estação').infer_objects(copy=False).interpolate(method='linear').dropna()
    df['Precipitação anterior'] = df['Precipitação dia'].shift(1); df = df.dropna()
    cols_X = ['Temperatura','Umidade','Velocidade Vento','Rajada Vento','Precipitação anterior']
    X, y = df[cols_X].values, df['Precipitação dia'].values
    X_scaled = StandardScaler().fit_transform(X); X_cnn = to_cnn1d_shape(X_scaled)

    splits = walk_forward_split(len(X_cnn), N_SPLITS)
    L = X_cnn.shape[2]; resultados = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, y_train = X_cnn[train_idx], y[train_idx]
        X_val, y_val = X_cnn[val_idx], y[val_idx]
        train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(WeatherDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

        for sparsity in SPARSITY_LEVELS:
            # Modelo base para aplicar poda
            model = build_cnn_model(length=L)
            conv1, conv2 = model[0], model[2]
            train_model(model, train_loader, EPOCHS_BASELINE, LR_BASELINE)

            # Aplica poda estruturada L1 e identifica filtros ativos
            prune.ln_structured(conv1, name="weight", amount=sparsity, n=1, dim=0)
            prune.ln_structured(conv2, name="weight", amount=sparsity, n=1, dim=0)
            mask1 = conv1.weight_mask.detach().cpu().numpy()[:,0,0]
            mask2 = conv2.weight_mask.detach().cpu().numpy()[:,0,0]
            kept1 = int(np.sum(mask1)); kept2 = int(np.sum(mask2))

            # Reconstrói o modelo com apenas os filtros restantes
            model_refactored = build_cnn_model(n_filters1=kept1, n_filters2=kept2, length=L).to(DEVICE)
            train_model(model_refactored, train_loader, EPOCHS_BASELINE, LR_BASELINE)

            ts = datetime.now().isoformat(timespec="seconds")
            r = evaluate(model_refactored, val_loader)
            r.update({"run": fold_id, "timestamp": ts, "sparsity": sparsity,
                      "filters1": kept1, "filters2": kept2})
            resultados.append(r)

    pd.DataFrame(resultados).to_csv(OUT_CSV, index=False, float_format="%.8f")
    print(f"\n✅ Resultados salvos em: {OUT_CSV}")

def train_model(model, loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

if __name__ == "__main__": main()
