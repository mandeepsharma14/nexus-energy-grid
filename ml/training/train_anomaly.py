"""
NexusGrid — Anomaly Detection Engine
Isolation Forest + Autoencoder ensemble for energy consumption anomaly detection.

Methods:
  - Isolation Forest (unsupervised, handles high-dimensional data well)
  - Autoencoder (deep learning, captures temporal patterns)
  - Statistical Process Control (for real-time streaming alerts)

© 2026 Mandeep Sharma. All rights reserved.
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/generated")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. AUTOENCODER DEFINITION
# ═══════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class EnergyAutoencoder(nn.Module):
        """
        LSTM Autoencoder for temporal anomaly detection.
        Learns normal energy patterns; high reconstruction error = anomaly.
        """

        def __init__(self, seq_len: int = 96, n_features: int = 6, encoding_dim: int = 32):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features

            # Encoder
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=encoding_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )

            # Decoder
            self.decoder = nn.LSTM(
                input_size=encoding_dim,
                hidden_size=n_features,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )

            self.output_layer = nn.Linear(n_features, n_features)

        def forward(self, x):
            # Encode
            _, (h_n, _) = self.encoder(x)
            # Repeat encoding across seq_len
            z = h_n[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
            # Decode
            decoded, _ = self.decoder(z)
            return self.output_layer(decoded)


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING FOR ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════
def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for anomaly detection."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_nighttime"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    # Rolling baselines
    df["expected_kw"] = (
        df.groupby(["hour", "day_of_week"])["kw"]
        .transform("median")
    )
    df["deviation_ratio"] = df["kw"] / (df["expected_kw"] + 1e-6)

    # Rate of change
    df["kw_delta"]   = df["kw"].diff().fillna(0)
    df["kw_delta_2"] = df["kw"].diff(2).fillna(0)

    # Z-score per hour
    df["kw_zscore"] = df.groupby("hour")["kw"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    df["rolling_24h_mean"] = df["kw"].rolling(96, min_periods=1).mean()
    df["rolling_24h_std"]  = df["kw"].rolling(96, min_periods=1).std().fillna(1)

    return df


# ═══════════════════════════════════════════════════════════
# 3. ISOLATION FOREST TRAINER
# ═══════════════════════════════════════════════════════════
def train_isolation_forest(df: pd.DataFrame, facility_id: str) -> IsolationForest:
    """Train Isolation Forest on normal operating data (exclude labeled anomalies)."""
    print(f"  Training Isolation Forest for {facility_id}...")

    feature_cols = [
        "kw", "deviation_ratio", "kw_delta", "kw_zscore",
        "hour", "day_of_week", "is_weekend", "is_nighttime",
        "rolling_24h_mean", "rolling_24h_std",
    ]

    # Use only non-anomaly data for training
    normal_df = df[~df["is_anomaly"]].dropna(subset=feature_cols)
    X = normal_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Contamination = expected anomaly fraction in unlabeled data
    model = IsolationForest(
        n_estimators=300,
        contamination=0.025,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        warm_start=False,
    )
    model.fit(X_scaled)

    # Save model + scaler
    with open(f"{MODEL_DIR}/iso_forest_{facility_id}.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_cols}, f)

    print(f"  ✓ Isolation Forest saved → iso_forest_{facility_id}.pkl")
    return model, scaler, feature_cols


# ═══════════════════════════════════════════════════════════
# 4. AUTOENCODER TRAINER
# ═══════════════════════════════════════════════════════════
def train_autoencoder(df: pd.DataFrame, facility_id: str, epochs: int = 20):
    if not TORCH_AVAILABLE:
        print(f"  [SKIP] PyTorch not available for Autoencoder on {facility_id}")
        return None

    print(f"  Training Autoencoder for {facility_id}...")

    feature_cols = [
        "kw", "deviation_ratio", "kw_delta",
        "kw_zscore", "rolling_24h_mean", "rolling_24h_std",
    ]

    normal_df = df[~df["is_anomaly"]].dropna(subset=feature_cols)
    X = normal_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    seq_len = 96  # 24h window
    sequences = []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])

    X_tensor = torch.tensor(np.array(sequences, dtype=np.float32))
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = EnergyAutoencoder(seq_len=seq_len, n_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — loss: {total_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), f"{MODEL_DIR}/autoencoder_{facility_id}.pt")
    with open(f"{MODEL_DIR}/ae_scaler_{facility_id}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"  ✓ Autoencoder saved → autoencoder_{facility_id}.pt")
    return model


# ═══════════════════════════════════════════════════════════
# 5. SCORING & THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════
def score_and_evaluate(
    df: pd.DataFrame,
    iso_model: IsolationForest,
    scaler: StandardScaler,
    feature_cols: list,
    facility_id: str,
) -> dict:
    """Score full dataset and evaluate against labeled anomalies."""
    df = df.copy().dropna(subset=feature_cols)
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    # Raw scores (more negative = more anomalous)
    scores = iso_model.score_samples(X_scaled)

    # Normalize to 0–1 (1 = most anomalous)
    s_min, s_max = scores.min(), scores.max()
    anomaly_score = 1 - (scores - s_min) / (s_max - s_min + 1e-9)

    df["anomaly_score"] = anomaly_score
    df["predicted_anomaly"] = (anomaly_score > 0.72).astype(int)
    df["true_anomaly"] = df["is_anomaly"].astype(int)

    # Metrics (only meaningful if labeled anomalies exist)
    labeled = df[df["true_anomaly"].isin([0, 1])]
    if len(labeled) > 10 and labeled["true_anomaly"].sum() > 0:
        precision = precision_score(labeled["true_anomaly"], labeled["predicted_anomaly"], zero_division=0)
        recall    = recall_score(labeled["true_anomaly"], labeled["predicted_anomaly"], zero_division=0)
        f1        = f1_score(labeled["true_anomaly"], labeled["predicted_anomaly"], zero_division=0)
        try:
            auc = roc_auc_score(labeled["true_anomaly"], labeled["anomaly_score"])
        except Exception:
            auc = 0.0
    else:
        precision = recall = f1 = auc = 0.0

    metrics = {
        "facility_id": facility_id,
        "total_records": len(df),
        "anomalies_detected": int(df["predicted_anomaly"].sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold": 0.72,
    }
    return metrics, df


# ═══════════════════════════════════════════════════════════
# 6. STATISTICAL PROCESS CONTROL (SPC)
# ═══════════════════════════════════════════════════════════
class SPCDetector:
    """
    Real-time SPC for streaming energy data.
    Uses CUSUM and 3-sigma control charts.
    """

    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k = k   # allowance parameter
        self.h = h   # decision interval
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.history = []

    def update(self, value: float) -> dict:
        self.history.append(value)
        if len(self.history) < 30:
            return {"alert": False, "cusum_pos": 0, "cusum_neg": 0}

        mu = np.mean(self.history[-96:])   # rolling 24h mean
        sigma = np.std(self.history[-96:]) + 1e-6

        z = (value - mu) / sigma
        self.cusum_pos = max(0, self.cusum_pos + z - self.k)
        self.cusum_neg = max(0, self.cusum_neg - z - self.k)

        alert_high = self.cusum_pos > self.h
        alert_low  = self.cusum_neg > self.h

        if alert_high or alert_low:
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0

        return {
            "alert": bool(alert_high or alert_low),
            "direction": "high" if alert_high else "low" if alert_low else "none",
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "zscore": round(z, 4),
        }


# ═══════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("NexusGrid — Anomaly Detection Engine Training")
    print("© 2026 Mandeep Sharma. All rights reserved.")
    print("=" * 60)

    df = pd.read_csv(f"{DATA_DIR}/interval_data.csv", parse_dates=["timestamp"])
    print(f"\nLoaded {len(df):,} interval records")

    all_metrics = []

    for facility_id in df["facility_id"].unique():
        print(f"\n── Facility: {facility_id} ──")
        fac_df = df[df["facility_id"] == facility_id].copy()
        fac_df = build_anomaly_features(fac_df)

        iso_model, scaler, feature_cols = train_isolation_forest(fac_df, facility_id)
        train_autoencoder(fac_df, facility_id, epochs=5)

        metrics, scored_df = score_and_evaluate(fac_df, iso_model, scaler, feature_cols, facility_id)
        all_metrics.append(metrics)

        print(f"  Evaluation → Anomalies: {metrics['anomalies_detected']}  "
              f"F1: {metrics['f1_score']:.3f}  AUC: {metrics['roc_auc']:.3f}")

        # Save scored results
        scored_df[["timestamp", "facility_id", "kw", "anomaly_score", "predicted_anomaly"]].to_csv(
            f"{MODEL_DIR}/anomaly_scores_{facility_id}.csv", index=False
        )

    # Save metrics summary
    pd.DataFrame(all_metrics).to_csv(f"{MODEL_DIR}/anomaly_detection_metrics.csv", index=False)
    print("\n" + "=" * 60)
    print("Anomaly detection training complete.")
    print(f"Models saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
