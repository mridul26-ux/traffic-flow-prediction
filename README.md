# Traffic Flow Prediction Using Road-Camera Footage (CNN-LSTM)

An Intelligent Transportation System (ITS) pipeline that ingests road-surveillance camera footage, detects and counts vehicles using a **Convolutional Neural Network (CNN)**, and forecasts short-term traffic flow and congestion states using a **Bidirectional Long Short-Term Memory (LSTM)** network with **Temporal Attention**.

---

## 🚀 Quick Start

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline (Training + Evaluation)
```bash
python train_pipeline.py
```
This automatically generates synthetic camera footage & benchmark time-series datasets, trains the CNN detector and LSTM forecaster, outputs evaluation metrics, and saves model checkpoints in `models/checkpoints/`.

### 3. Run the Live Video Simulation & Forecasting Demo
```bash
python demo.py
```
Processes 20 simulated camera frames, performs vehicle detection, updates the temporal feature buffer, and renders a live HUD overlay to `reports/figures/demo_frame.png` and an animated video to `reports/figures/traffic_simulation.mp4`.

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## 📁 Repository Structure
```
Traffic flow/
├── data/
│   └── raw/
│       ├── camera_dataset_metadata.csv  # Metadata for 250 camera frames
│       ├── frames/                      # Road-camera frames (224x224)
│       └── traffic_timeseries.csv       # 6,048 intervals of 5-min traffic time-series
├── models/
│   ├── cnn_detector.py                  # 4-block CNN for vehicle density & feature extraction
│   ├── lstm_predictor.py                # Bidirectional LSTM with Temporal Attention
│   ├── cnn_lstm_pipeline.py             # Integrated end-to-end inference pipeline
│   └── checkpoints/
│       ├── traffic_cnn.pt               # Trained CNN PyTorch weights
│       └── traffic_lstm.pt              # Trained LSTM PyTorch weights
├── utils/
│   ├── dataset_generator.py             # Realistic camera frame & time-series synthesizer
│   ├── preprocessor.py                  # Transforms, data augmentations & sliding window framing
│   ├── metrics.py                       # MAE, RMSE, MAPE, R2, F1-Score & Confusion Matrix
│   └── visualizer.py                    # Training curves, confusion matrices, forecast plots
├── reports/
│   ├── cnn_metrics.json                 # Test metrics for CNN
│   ├── lstm_metrics.json                # Test metrics for LSTM
│   └── figures/
│       ├── cnn_loss.png                 # CNN training & validation loss curve
│       ├── cnn_confusion_matrix.png     # CNN congestion classification confusion matrix
│       ├── lstm_loss.png                # LSTM training & validation loss curve
│       ├── lstm_confusion_matrix.png    # LSTM congestion classification confusion matrix
│       ├── flow_prediction_curve.png    # Actual vs predicted traffic flow comparison
│       ├── demo_frame.png               # Live HUD annotated frame
│       └── traffic_simulation.mp4       # Real-time traffic camera video demo
├── train_cnn.py                         # Standalone CNN trainer & evaluator
├── train_lstm.py                        # Standalone LSTM trainer & evaluator
├── train_pipeline.py                    # Master project execution pipeline
├── demo.py                              # Live interactive video simulation
├── app.py                               # Streamlit web dashboard
└── requirements.txt                     # Python package dependencies
```

---

## 📊 Summary of Model Performance

### CNN Vehicle Density & Congestion Model
- **Vehicle Count MAE:** `0.815` vehicles per frame
- **Vehicle Count RMSE:** `1.090`
- **Count $R^2$ Score:** `0.9593` (explains 95.9% of variance)
- **Congestion Classification Accuracy:** `65.79%`

### LSTM Traffic Flow Forecaster (Multi-Horizon)
- **$t+5$ min Forecast:** MAE = `7.35` veh/int, RMSE = `9.44`, $R^2$ = `0.9368`
- **$t+10$ min Forecast:** MAE = `7.32` veh/int, RMSE = `9.58`, $R^2$ = `0.9350`
- **$t+15$ min Forecast:** MAE = `7.47` veh/int, RMSE = `9.89`, $R^2$ = `0.9308`
- **Congestion State Accuracy:** `87.20%`
- **Weighted F1-Score:** `0.8608`
