import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# fix path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from petromind.pipeline.features import SequenceFeatureExtractor
from petromind.pipeline.lstm_model import LSTMClassifier

print("START TEST")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# LOAD MODEL
# =========================
mean = np.load("mean.npy")
std  = np.load("std.npy")

dummy      = np.zeros((1, 30, 24), dtype=np.float32)
dummy_feat = SequenceFeatureExtractor().transform(dummy)
input_dim  = dummy_feat.shape[2]

model = LSTMClassifier(input_dim=input_dim).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
print(f"Input dim: {input_dim}")

# =========================
# LOAD TEST DATA
# =========================
test_path  = r"D:\petromind\PetroMind\Prediction_Analysis_Results\02_Data\Raw\All_test_data.xlsx"
all_sheets = pd.read_excel(test_path, sheet_name=None)

frames = []
uid_offset = 0

for name, df in all_sheets.items():
    if "unit id" in df.columns:
        df = df.rename(columns={"unit id": "unit_id"})
    df = df.copy()
    df["unit_id"] += uid_offset
    uid_offset = df["unit_id"].max()
    frames.append(df)

df_test = pd.concat(frames, ignore_index=True)
df_test = df_test.sort_values(["unit_id", "cycle"])
print(f"Test engines: {df_test['unit_id'].nunique()}")

# =========================
# WINDOWING
# =========================
WINDOW_SIZE  = 30
feature_cols = [c for c in df_test.columns
                if c not in ["unit_id", "cycle", "dataset"]]

all_engines   = sorted(df_test["unit_id"].unique())
valid_engines = set()
X_test        = []

for uid in all_engines:
    vals = df_test[df_test["unit_id"] == uid][feature_cols].values
    if len(vals) >= WINDOW_SIZE:
        X_test.append(vals[-WINDOW_SIZE:])
        valid_engines.add(uid)

X_test = np.array(X_test, dtype=np.float32)
print(f"Test windows: {X_test.shape}")

# =========================
# FEATURES
# =========================
extractor = SequenceFeatureExtractor()
X_test    = extractor.transform(X_test)
print(f"After features: {X_test.shape}")

# =========================
# NORMALIZATION
# =========================
X_test = (X_test - mean) / std

# =========================
# PREDICTION
# =========================
loader    = DataLoader(torch.tensor(X_test, dtype=torch.float32), batch_size=64)
all_probs = []

with torch.no_grad():
    for X_batch in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs   = torch.softmax(outputs, dim=1)
        all_probs.extend(probs[:, 1].cpu().numpy())

all_probs = np.array(all_probs)
preds     = (all_probs > 0.4).astype(int)
print(f"Predictions: {len(preds)}")

# =========================
# LOAD RUL
# =========================
rul_path   = r"D:\petromind\PetroMind\Prediction_Analysis_Results\02_Data\Processed\RUL_data.xlsx"
rul_sheets = pd.read_excel(rul_path, sheet_name=None)

rul_list = []
for name, df in rul_sheets.items():
    rul_list.extend(df.values.flatten())

rul_values    = np.array(rul_list)
valid_indices = [i for i, uid in enumerate(all_engines)
                 if uid in valid_engines]
y_test        = (rul_values[valid_indices] <= 30).astype(int)

print(f"True At Risk : {y_test.sum()} ({y_test.mean():.1%})")
print(f"Pred At Risk : {preds.sum()} ({preds.mean():.1%})")

# =========================
# EVALUATION
# =========================
print("\nRESULT")
print(classification_report(y_test, preds,
      target_names=['Healthy', 'At Risk']))
print(f"ROC-AUC: {roc_auc_score(y_test, all_probs):.4f}")