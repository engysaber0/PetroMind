"""
PetroMind — Predictive Maintenance ML Pipeline

Modules:
    pipeline.config      - Pipeline configuration dataclass
    pipeline.utils       - Data loading, validation, edge-case handling
    pipeline.labeling    - RUL computation and binary classification labels
    pipeline.windowing   - Sliding-window construction (no future leakage)
    pipeline.features    - Statistical, signal, health-indicator, and sensor-fusion features
    pipeline.dataset     - PyTorch Dataset / DataLoader with time-based splits
"""
