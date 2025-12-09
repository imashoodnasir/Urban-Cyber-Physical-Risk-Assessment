# Spatio–Temporal Multimodal Urban Cyber–Physical Risk Assessment

This repository provides a reference implementation for the paper:

**Spatio–Temporal Deep Learning on Multimodal Remote Sensing Data for Urban Cyber–Physical Risk Assessment**

The code demonstrates how multimodal Earth observation (EO) data and auxiliary urban information can be fused using cross‑modal attention and spatio‑temporal Transformers to generate calibrated urban vulnerability maps with uncertainty estimation.

---

## Method Overview

The implemented pipeline follows seven major steps:

1. **Multimodal Data Harmonization**
   - Ingest PlanetScope RGB, Sentinel‑1 SAR, Sentinel‑2 multispectral, nighttime lights, population grids, traffic sensors, and OSM infrastructure graphs.
   - Reproject all data sources onto a common spatial grid and temporally align them into a unified tensor:
     \
     `X(q,t) ∈ R^{T × C × H × W}`.

2. **Modality‑Specific Spatial Encoding**
   - CNN encoders extract raster features (optical, SAR, NTL, population, traffic).
   - A lightweight GCN placeholder embeds OSM graph structures.

3. **Cross‑Modal Attention Fusion**
   - Q/K/V projections are computed per modality.
   - Dot‑product cross‑attention assigns adaptive modality weights.
   - Fused representation: `F_fuse(q,t)`.

4. **Spatio‑Temporal Modeling**
   - 1D temporal convolutions capture short‑term fluctuations.
   - Transformer encoder models long‑range temporal dependencies.
   - Output latent sequence: `H(q,t)`.

5. **Gated Risk Embedding**
   - Sigmoid gates `η(q,t)` perform temporal pooling.
   - Aggregated embedding `R̃(q)` captures abnormal activity patterns.

6. **Prediction and Calibration**
   - **Regression head:** continuous vulnerability scores.
   - **Classification head:** low / medium / high risk categories.
   - **Monte‑Carlo Dropout:** repeated stochastic passes estimate prediction uncertainty.

7. **Deployment and Mapping**
   - Model outputs are reprojected back to GIS coordinates.
   - City‑scale vulnerability, risk‑class, and uncertainty maps are generated.

---

## Repository Structure

```
urban_risk_model/
├── README.md
├── config.py
├── datasets.py
├── train.py
├── inference.py
├── utils.py
└── models/
    ├── harmonization.py
    ├── encoders.py
    ├── fusion.py
    ├── temporal.py
    ├── risk_head.py
    └── full_model.py
```

- **config.py** – training and model hyperparameters  
- **datasets.py** – toy dataset generator (replace with real harmonized EO data)  
- **models/** – all deep-learning modules  
- **train.py** – supervised training loop  
- **inference.py** – Monte‑Carlo dropout inference and uncertainty estimation  
- **utils.py** – metric utilities

---

## Installation

Create environment and install dependencies:

```bash
conda create -n urban-risk python=3.10
conda activate urban-risk

pip install torch torchvision numpy matplotlib tqdm
```

CUDA users may install PyTorch according to their GPU drivers.

---

## Quick Start (Synthetic Demo)

Train on a randomly generated multimodal dataset:

```bash
python train.py --epochs 5 --num-samples 64
```

Run uncertainty‑aware inference:

```bash
python inference.py --num-samples 16 --checkpoint model.pt
```

---

## Adapting to Real Data

1. Replace the dataset in `datasets.py` with real harmonized EO tensors:
   - Each sample should be shaped as `(T, C, H, W)`.

2. Implement the real preprocessing pipeline in:
   - `models/harmonization.py`

3. Optionally separate modality channels and pass them to individual encoders inside
   `models/encoders.py`.

4. Add evaluation metrics (SCOT score, IoU, F1) to `utils.py` if required.

---

## Notes

- This implementation prioritizes **research clarity** over full production optimization.
- Graph embedding (GNN) is provided as a lightweight placeholder and should be extended for detailed OSM topology modeling.
- All uncertainty estimates use standard Monte‑Carlo dropout approximations.

---

## License

This repository is provided for academic and research use only.
