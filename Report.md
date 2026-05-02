# LSTM Model – Detailed Technical Report

## 1. Objective

The objective of this project is to build an LSTM-based model to classify whether an engine is at risk of failure using time-series sensor data.

---

## 2. Data Preparation

* Loaded multiple datasets from Excel sheets
* Combined them into a single dataset
* Ensured unique engine IDs across all datasets
* Sorted data by engine and cycle

---

## 3. RUL Computation & Labeling

* RUL (Remaining Useful Life) was calculated as:

  RUL = max_cycle - current_cycle

* RUL values were clipped at 125 to reduce extreme values

* Converted into classification:

  * At Risk (1): RUL ≤ 30
  * Healthy (0): RUL > 30

---

## 4. Sliding Window Approach

* Window size = 30 cycles
* Created sequences using sliding windows
* Label of each window = label of last timestep

This allows the model to learn temporal patterns.

---

## 5. Feature Engineering

Additional features were extracted from sequences:

* First-order difference (trend over time)
* Rolling mean (smoothing)
* Global trend (slope across window)

Final features:

* Original features + engineered features
* Increased feature richness significantly

---

## 6. Train / Validation Split

* Split was done **by engine ID (not randomly)**
* Ensures no data leakage between train and validation

---

## 7. Normalization

* Mean and standard deviation computed from training data only
* Applied same normalization to validation and test sets

---

## 8. Model Architecture

* Model: LSTM (PyTorch)
* Input size: 96 features
* Hidden size: 128
* Number of layers: 2
* Dropout: 0.1
* Output: Binary classification (2 classes)

---

## 9. Handling Class Imbalance

* Dataset had imbalance between healthy and at-risk samples

* Used weighted CrossEntropy:

  weight = count_healthy / count_at_risk

* Helped model focus on minority class

---

## 10. Training Strategy

* Optimizer: Adam
* Learning rate: 5e-4
* Weight decay used (regularization)

### Overfitting Handling:

* Used Dropout inside LSTM
* Used Early Stopping (patience = 8)
* Monitored validation loss and F1 score

---

## 11. Model Selection

* Best model saved based on **highest F1 score**
* This ensures better balance between precision and recall

---

## 12. Evaluation (Validation)

* Used classification report
* Also tested custom threshold = 0.4 instead of default 0.5

This improved detection of "At Risk" class

---

## 13. Test Strategy (Important Insight)

For test data:

* Only the **last 30 cycles (last window)** of each engine were used

### Reason:

* Represents the most recent engine condition
* Matches real-world prediction scenario

### Impact:

* Improved prediction accuracy
* Focused on failure-relevant signals

---

## 14. Final Results

* Accuracy: ~0.94 – 0.96
* F1 Score: ~0.88 – 0.91
* ROC-AUC: ~0.99

Model shows strong ability to detect at-risk engines.

---

## 15. Key Insights

* Feature engineering significantly improved performance
* Using last window for test gave better real-world predictions
* Handling class imbalance was critical
* Overfitting was controlled using dropout + early stopping

---
## Design Decision: Error Trade-off

During experimentation, different decision thresholds were tested.

A lower threshold (0.4) was selected as it improved the model’s ability to detect at-risk engines.

This resulted in:

* Higher recall (better detection of failures)
* Slight increase in false positives

### Justification:

In predictive maintenance, missing a real failure is more critical than raising a false alarm.

Therefore, it is preferable for the model to occasionally classify a healthy engine as "At Risk" rather than failing to detect an actual failure.





