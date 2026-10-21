# DeepSentinel Inspection & Defense Scripts

This directory contains the primary demonstration and formula verification scripts for your thesis defense and reviewer inspection.

### Documentation & Reference Cheat Sheet
For full formulas, step-by-step arithmetic derivations, dataset origins, and model dissection, see:
**[docs/PIPELINE_DEFENSE_CHEATSHEET.md](file:///d:/Documents/Programming/Thesis_G10/docs/PIPELINE_DEFENSE_CHEATSHEET.md)**

---

### Available Scripts in this Folder

#### 1. `inspect_pipeline.py` — Raw Formula & Evaluation Inspector
Run from repository root (`d:\Documents\Programming\Thesis_G10`):
```powershell
# Inspect raw evaluation contingency and formula derivations (Accuracy, Precision, Recall, F1, AUC)
python scripts/inspection/inspect_pipeline.py --part evaluation

# Inspect Compact Bilinear Pooling (Count sketch, FFT, power norm, unit L2 norm = 1.000000)
python scripts/inspection/inspect_pipeline.py --part cbp

# Inspect 299D multimodal bottleneck architecture & Softmax distributions
python scripts/inspection/inspect_pipeline.py --part model

# Inspect Multi-Task loss formulations (Focal loss, Masked cross entropy, Margin)
python scripts/inspection/inspect_pipeline.py --part losses

# Inspect paired DeLong AUC test & bootstrap scorecard vs ACE-Net
python scripts/inspection/inspect_pipeline.py --part significance

# Run all modules sequentially
python scripts/inspection/inspect_pipeline.py --part all
```

#### 2. `demo_training_flow.py` — Live End-to-End Training Flow Demonstration
Simulates a training step from scratch on 8 demo clips (4 Real + 4 Fake) in ~4 seconds:
```powershell
# Run full training lifecycle (Ingestion -> CBP -> Discrepancy -> 299D -> Loss -> Backward -> Step -> Val)
python scripts/inspection/demo_training_flow.py

# Run individual stages on demand:
python scripts/inspection/demo_training_flow.py --stage ingestion
python scripts/inspection/demo_training_flow.py --stage forward
python scripts/inspection/demo_training_flow.py --stage loss
python scripts/inspection/demo_training_flow.py --stage backward
python scripts/inspection/demo_training_flow.py --stage validation

# Scale to 16 demo clips
python scripts/inspection/demo_training_flow.py --clips 16
```
