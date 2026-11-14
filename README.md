# Short-Term Traffic Flow Prediction from Road Camera Footage
**Computer Vision & Time-Series Forecasting Course Project**  
*Author: Mridul | Department of Computer Science & Engineering*

---

## Abstract
Traffic congestion in metropolitan road networks causes significant economic loss, fuel wastage, and transit delays. While inductive loop sensors are historically used for traffic monitoring, they require disruptive road installation and frequent maintenance. This project implements a camera-based intelligent traffic monitoring system that pairs:
1. A **Convolutional Neural Network (CNN)** for vehicle detection, density estimation, and visual feature extraction from roadside camera frames.
2. A **Bidirectional Long Short-Term Memory (BiLSTM)** network with temporal attention to predict short-term vehicle flow ($t+5$, $t+10$, and $t+15$ minutes) and classify upcoming congestion levels.

---

## System Overview

```
[ Camera Video Stream ]
          │
          ▼
┌─────────────────────────────────┐
│       CNN Feature Extractor     │  --> Frame Vehicle Count
│   (Conv2d -> BatchNorm -> Pool) │  --> Visual Spatial Embeddings (64-dim)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│     Rolling Temporal Buffer     │  --> Past 60-min history (12 intervals)
│ [Flow, Speed, Occupancy, Time]  │
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│   Bidirectional LSTM + Attention│
│  (Temporal sequence processing) │
└─────────────────────────────────┘
          │
          ▼
[ Multi-Step Forecast: +5m, +10m, +15m & Congestion Alert ]
```

---

## 1. Problem Definition
Municipal traffic management systems need proactive, real-time visibility into traffic conditions to dynamically adjust signal light timings, manage highway on-ramp metering, and alert emergency services. The objective of this project is to build an end-to-end software pipeline capable of converting raw road-camera video into quantitative short-term traffic volume predictions without requiring embedded road hardware.

---

## 2. Dataset

### Visual Data (Road-Camera Surveillance)
- **Samples:** 250 annotated road surveillance frames captured at 1080p perspective and resized to $224 \times 224 \times 3$.
- **Classes:** Vehicles are labeled across 4 categories:
  - Sedan / Compact Car (~60%)
  - SUV / Van (~20%)
  - Truck / Bus (~12%)
  - Motorcycle (~8%)
- **Congestion States:** Frames are labeled into 4 operational regimes based on density: Free Flow, Moderate, Heavy, and Gridlock.

### Sequential Time-Series Data
- **Horizon:** 21 days of continuous traffic measurements recorded in 5-minute aggregation intervals (total of 6,048 sequential steps).
- **Features Tracked:**
  - Vehicle Flow rate (vehicles per 5-minute interval)
  - Average spot speed ($\text{km/h}$)
  - Road occupancy rate ($\%$)
  - Time identifiers (hour of day, day of week, weekend indicator)

---

## 3. Data Preprocessing & Augmentation

1. **Image Transformations:**
   - Standardized resizing to $224 \times 224$.
   - Normalized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - Training augmentations: Random horizontal flipping ($p=0.5$), photometric color jitter (brightness, contrast, saturation $\pm 0.2$).
2. **Temporal Feature Encoding:**
   - Time variables mapped to cyclical sine/cosine coordinates to preserve day-night continuity:
     $$\sin(2\pi \cdot \text{hour}/24), \quad \cos(2\pi \cdot \text{hour}/24)$$
3. **Sequence Framing & Scaling:**
   - Standardized using `StandardScaler` fitted strictly on the training partition to avoid lookahead bias.
   - Chronological sliding window of $12$ past steps ($60$ minutes) predicting $3$ future intervals ($15$ minutes ahead).
4. **Data Splits:** Chronological $70\%$ train, $15\%$ validation, and $15\%$ test split.

---

## 4. Model Architecture

### CNN Density Estimator (`models/cnn_detector.py`)
- 4 convolutional blocks with $3\times3$ kernels, batch normalization, ReLU activations, and $2\times2$ max-pooling layers.
- Global adaptive average pooling layer followed by a dense bottleneck projecting to a 64-dimensional latent embedding.
- Dual output heads:
  - Regression head for continuous vehicle count per frame.
  - Classification head (Softmax) for 4-class congestion estimation.

### BiLSTM Temporal Forecaster (`models/lstm_predictor.py`)
- Input linear projection mapping 7-dimensional features to hidden dimensions.
- 2-layer Bidirectional LSTM ($h=64$) capturing forward and backward temporal dependencies.
- Temporal attention mechanism that learns attention weights $\alpha_t$ across the 12-step sequence.
- Dual output heads:
  - Linear regressor predicting flow for horizons $t+1$, $t+2$, and $t+3$.
  - Softmax classifier predicting upcoming congestion regime.

---

## 5. Training Settings

- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$).
- **Learning Rate:** $10^{-3}$ (CNN) and $2 \times 10^{-3}$ (LSTM), controlled by `ReduceLROnPlateau` (decay factor $0.5$, patience $3$).
- **Loss Functions:**
  - CNN: Smooth L1 Loss (Count) $+ 0.6 \times$ Cross Entropy Loss (Congestion).
  - LSTM: Mean Squared Error (Flow) $+ 0.4 \times$ Cross Entropy Loss (Congestion).
- **Batch Size:** 16 (CNN), 32 (LSTM).
- **Gradient Clipping:** Max norm $5.0$ to prevent exploding gradients in the recurrent layers.

---

## 6. Experimental Results & Evaluation

The models were evaluated on the held-out test datasets:

### CNN Vehicle Detection
| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | 0.815 vehicles |
| Root Mean Squared Error (RMSE) | 1.090 |
| Mean Absolute Percentage Error (MAPE) | 10.27% |
| Coefficient of Determination ($R^2$) | 0.9593 |
| Congestion Classification Accuracy | 65.79% |

### LSTM Traffic Flow Multi-Step Forecasting
| Forecast Horizon | MAE (veh/5-min) | RMSE | MAPE (%) | $R^2$ Score |
|---|---|---|---|---|
| **$t+5$ min** | 7.35 | 9.44 | 22.41% | **0.9368** |
| **$t+10$ min** | 7.32 | 9.58 | 22.72% | **0.9350** |
| **$t+15$ min** | 7.47 | 9.89 | 22.75% | **0.9308** |

### Congestion Regime Classification (Test Partition)
- **Accuracy:** 87.20%
- **Weighted Precision:** 0.8564
- **Weighted Recall:** 0.8720
- **Weighted F1-Score:** 0.8608

---

## 7. Analysis & Discussion

- **Strengths:** The model achieves high explanatory power ($R^2 > 0.93$) across all prediction horizons, demonstrating strong tracking of diurnal morning and evening peak traffic. The temporal attention layer quickly responds to sudden speed drops.
- **Error Modes:** Model variance increases slightly during extreme, abrupt traffic accidents due to the relative scarcity of severe gridlock samples in the training split.
- **Physical Limitations:** Single-camera setups remain susceptible to optical occlusion (e.g. large trucks blocking smaller vehicles in distant lanes) and severe glare.

---

## 8. Future Work
- Implementing spatial-temporal graph neural networks (ST-GCN) to link multiple intersections across a road network.
- Deploying the model onto edge inference hardware (such as NVIDIA Jetson) using TensorRT optimization.
- Adding domain adaptation layers for adverse weather (rain, night glare, dense fog).

---

## 9. Setup & Execution

### Local Python Environment
```bash
# Install dependencies
pip install -r requirements-train.txt

# Train and evaluate both models
python train_pipeline.py

# Run interactive simulation demo
python demo.py

# Launch local dashboard
streamlit run app.py
```

### Web Deployment
The repository includes a serverless web version with an interactive canvas simulation and live forecast chart ready for deployment on Vercel:
- Frontend: `index.html`
- API handler: `api/index.py`
- Configuration: `vercel.json`
