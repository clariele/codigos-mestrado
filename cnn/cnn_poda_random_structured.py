# cnn_poda_random_structured.py
# ============================================================
# CNN 1D (PyTorch) — PODA ESTRUTURADA RANDOM
# Walk-forward + medição de energia (INA219) + memória (psutil)
# Sparsity levels: 20%, 40%, 60%, 80%
# ============================================================

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune

import threading
from collections import deque
import board
import busio
from adafruit_ina219 import INA219
import psutil

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
OUT_CSV         = "resultados_cnn_poda_random_structured.csv"

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

# ----------------------- Modelo -----------------------
class CNNRegressor(nn.Module):
    def __init__(self, in_channels=1, length=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.fc1   = nn.Linear(16 * length, 64)
        self.fc2   = nn.Linear(64, 1)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU(),
            nn.Flatten(),
            self.fc1, nn.ReLU(),
            self.fc2
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ----------------------- Treino / Avaliação -----------------------
def train_model(model, loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model_with_energy(model, loader, label="Avaliação (energia)"):
    model.eval()
    y_true, y_pred = [], []
    meter = EnergyMeter(sample_interval=SAMPLE_INTERVAL, addr=INA_ADDR)

    with torch.no_grad():
        for xb, _ in loader: _ = model(xb); break

    mem_mb_before_energy_infer = get_memory_usage_mb()
    print(f"💾 Memória antes da inferência: {mem_mb_before_energy_infer:.2f} MB")

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
    _mae, _rmse, _r2 = mae(y_true, y_pred), rmse(y_true, y_pred), r2_score_np(y_true, y_pred)
    total_time = t1 - t0
    time_per_sample_ms = (total_time / len(loader.dataset)) * 1000.0

    stats = meter.summarize() if meter.enabled else {}
    energy_uJ_per_inf = (stats["energy_J"] * 1e6 / n_preds) if stats.get("energy_J") else None

    return {
        "mae": _mae, "rmse": _rmse, "r2": _r2,
        "total_inference_s": total_time, "time_per_sample_ms": time_per_sample_ms,
        "duration_s": stats.get("duration_s"), "energy_J": stats.get("energy_J"),
        "energy_uJ_per_inf": energy_uJ_per_inf,
        "avg_power_W": stats.get("avg_power_W"), "peak_power_W": stats.get("peak_power_W"),
        "avg_current_mA": stats.get("avg_current_mA"), "peak_current_mA": stats.get("peak_current_mA"),
        "avg_voltage_V": stats.get("avg_voltage_V"), "n_samples": stats.get("n_samples"),
        "mem_mb_before_energy_infer": mem_mb_before_energy_infer
    }

# ----------------------- Medição de Energia -----------------------
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

# ----------------------- Walk-forward -----------------------
def walk_forward_split(n, n_splits=5):
    fold_size = n // (n_splits+1); splits=[]
    for i in range(n_splits):
        end_train = fold_size*(i+1); start_val = end_train; end_val = start_val+fold_size
        if end_val>n: break
        splits.append((range(0,end_train),range(start_val,end_val)))
    return splits, fold_size

# ----------------------- Main -----------------------
def main():
    set_seed(SEED)
    df = pd.read_csv(CSV_PATH)
    df = df[df['Nome'].str.strip()=="Estação Pesca - UFRPE"]
    df['Data estação'] = pd.to_datetime(df['Data estação'], errors='coerce')
    df = df.sort_values('Data estação').infer_objects(copy=False).interpolate(method='linear').dropna()
    df['Precipitação anterior'] = df['Precipitação dia'].shift(1); df = df.dropna()
    cols_X = ['Temperatura','Umidade','Velocidade Vento','Rajada Vento','Precipitação anterior']
    X,y = df[cols_X].values, df['Precipitação dia'].values
    X_scaled = StandardScaler().fit_transform(X); X_cnn = to_cnn1d_shape(X_scaled)

    splits,_ = walk_forward_split(len(X_cnn),N_SPLITS)
    resultados=[]; L = X_cnn.shape[2]

    for fold_id,(train_idx,val_idx) in enumerate(splits,1):
        X_train,y_train = X_cnn[train_idx],y[train_idx]
        X_val,y_val = X_cnn[val_idx],y[val_idx]
        train_loader = DataLoader(WeatherDataset(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
        val_loader = DataLoader(WeatherDataset(X_val,y_val),batch_size=BATCH_SIZE,shuffle=False)

        for sparsity in SPARSITY_LEVELS:
            model = CNNRegressor(length=L).to(DEVICE)
            train_model(model,train_loader,EPOCHS_BASELINE,LR_BASELINE)

            # aplicar poda estruturada random nas Conv1d
            prune.random_structured(model.conv1, name="weight", amount=sparsity, dim=0)
            prune.random_structured(model.conv2, name="weight", amount=sparsity, dim=0)

            mem_mb_after_baseline = get_memory_usage_mb()
            print(f"💾 Memória após treino baseline (sparsity {sparsity}): {mem_mb_after_baseline:.2f} MB")

            ts = datetime.now().isoformat(timespec="seconds")
            r = evaluate_model_with_energy(model,val_loader,label=f"Fold {fold_id} (sparsity {sparsity})")
            r.update({"run":fold_id,"timestamp":ts,"mem_mb_after_baseline":mem_mb_after_baseline,"sparsity":sparsity})
            resultados.append(r)

    df_out = pd.DataFrame(resultados,columns=[
        "run","timestamp","sparsity","mae","rmse","r2",
        "total_inference_s","time_per_sample_ms",
        "duration_s","energy_J","energy_uJ_per_inf",
        "avg_power_W","peak_power_W","avg_current_mA","peak_current_mA",
        "avg_voltage_V","n_samples",
        "mem_mb_after_baseline","mem_mb_before_energy_infer"
    ])
    df_out.to_csv(OUT_CSV,index=False,float_format="%.8f")
    print(f"\n✅ Resultados salvos em: {OUT_CSV}")

if __name__=="__main__": main()
