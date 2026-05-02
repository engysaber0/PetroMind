import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== IMPORT PIPELINE =====
from petromind.pipeline.config import PipelineConfig
from petromind.pipeline.labeling import compute_rul, compute_classification_label
from petromind.pipeline.windowing import build_sliding_windows
from petromind.pipeline.features import SequenceFeatureExtractor
from petromind.pipeline.lstm_model import LSTMClassifier

print("START TRAINING ")

# =========================
# STEP 1: LOAD DATA
# =========================
file_path = r"D:\petromind\PetroMind\Prediction_Analysis_Results\02_Data\Raw\All_train_data.xlsx"

all_sheets = pd.read_excel(file_path, sheet_name=None)

frames = []
uid_offset = 0

for name, df in all_sheets.items():
    if "unit id" in df.columns:
        df = df.rename(columns={"unit id": "unit_id"})
    df = df.copy()
    df["unit_id"] = df["unit_id"] + uid_offset
    uid_offset += df["unit_id"].max()
    frames.append(df)

df = pd.concat(frames, ignore_index=True)
print(f"Merged shape   : {df.shape}")
print(f"Unique engines : {df['unit_id'].nunique()}")

# =========================
# STEP 2: LABELING
# =========================
cfg = PipelineConfig()

df = compute_rul(df, cfg)
df = compute_classification_label(df, cfg)

print(f"Label distribution:")
print(pd.Series(df['label']).value_counts(normalize=True))

# =========================
# STEP 3: WINDOWING
# =========================
X, y_cls, y_rul, engine_ids = build_sliding_windows(df, cfg)
print(f"X shape: {X.shape}")

# =========================
# STEP 4: FEATURE ENGINEERING
# =========================
extractor = SequenceFeatureExtractor(window_size=cfg.window_size)
X = extractor.transform(X)
print(f"After features: {X.shape}")

# =========================
# STEP 5: SPLIT (by engine)
# =========================
unique_engines = np.sort(np.unique(engine_ids))
n_val          = int(len(unique_engines) * 0.2)
val_engines    = set(unique_engines[-n_val:])

train_mask = np.array([eid not in val_engines for eid in engine_ids])
val_mask   = ~train_mask

X_train, y_train = X[train_mask], y_cls[train_mask]
X_val,   y_val   = X[val_mask],   y_cls[val_mask]

print(f"Train: {X_train.shape} | Positive: {y_train.mean():.1%}")
print(f"Val  : {X_val.shape}   | Positive: {y_val.mean():.1%}")

# =========================
# STEP 6: NORMALIZATION
# =========================
mean = X_train.mean(axis=(0, 1))
std  = X_train.std(axis=(0, 1)) + 1e-8

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

print("Normalized")

# =========================
# STEP 7: DATA LOADER
# =========================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=64, shuffle=True, generator=g
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=64, shuffle=False
)

# =========================
# STEP 8: MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)

# Class weights
class_counts  = np.bincount(y_train)
scale         = class_counts[0] / class_counts[1]
class_weights = torch.tensor([1.0, scale], dtype=torch.float32).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

print(f"Scale pos weight: {scale:.2f}")

# =========================
# STEP 9: TRAIN
# =========================
best_f1, best_val_loss   = 0, float('inf')
patience, patience_counter = 8, 0

for epoch in range(50):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output    = model(X_batch)
            val_loss += criterion(output, y_batch).item()
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "model.pth")

    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {f1:.4f}")

print(f"\nBest F1: {best_f1:.4f}")

# =========================
# STEP 10: SAVE
# =========================
np.save("mean.npy", mean)
np.save("std.npy",  std)
print("MODEL & SCALER SAVED")