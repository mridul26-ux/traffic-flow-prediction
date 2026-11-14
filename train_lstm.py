"""
train_lstm.py
Training and Evaluation Script for Temporal Traffic Flow Prediction using LSTM.
Trains on sequential sliding windows, evaluates multi-step forecasting and congestion states,
saves model checkpoint and publication-ready evaluation charts.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.lstm_predictor import TrafficLSTM
from utils.dataset_generator import generate_traffic_timeseries, CONGESTION_LEVELS
from utils.preprocessor import prepare_timeseries_data
from utils.metrics import calculate_regression_metrics, calculate_classification_metrics
from utils.visualizer import plot_loss_curves, plot_confusion_matrix, plot_traffic_flow_prediction


def train_lstm(data_csv="data/raw/traffic_timeseries.csv", seq_len=12, forecast_horizon=3,
               epochs=20, batch_size=32, lr=2e-3, device="cpu"):
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    device = torch.device(device)
    print(f"[Train LSTM] Using device: {device}")
    
    # Ensure timeseries dataset exists
    if not os.path.exists(data_csv):
        print("[Train LSTM] Generating synthetic traffic time-series data (21 days)...")
        generate_traffic_timeseries(num_days=21, output_csv=data_csv)
        
    datasets, scaler, feat_cols = prepare_timeseries_data(
        data_csv, seq_len=seq_len, forecast_horizon=forecast_horizon, train_ratio=0.70, val_ratio=0.15
    )
    
    print(f"[Train LSTM] Feature columns ({len(feat_cols)}): {feat_cols}")
    print(f"[Train LSTM] Sequence Length: {seq_len} intervals ({seq_len * 5} mins history)")
    print(f"[Train LSTM] Forecast Horizon: {forecast_horizon} intervals ({forecast_horizon * 5} mins ahead)")
    print(f"[Train LSTM] Dataset Split: Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}")
    
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    
    model = TrafficLSTM(
        input_dim=len(feat_cols),
        hidden_dim=64,
        num_layers=2,
        forecast_horizon=forecast_horizon,
        num_classes=4,
        dropout=0.2
    ).to(device)
    
    criterion_flow = nn.MSELoss()
    criterion_cong = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    checkpoint_path = "models/checkpoints/traffic_lstm.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for x, y_flow, y_cong in train_loader:
            x = x.to(device)
            y_flow = y_flow.to(device)
            y_cong = y_cong.to(device)
            
            optimizer.zero_grad()
            pred_flow, pred_cong, _ = model(x)
            
            # Scaled flow loss normalization
            flow_loss = criterion_flow(pred_flow / 100.0, y_flow / 100.0) * 100.0
            cong_loss = criterion_cong(pred_cong, y_cong)
            loss = flow_loss + 0.4 * cong_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * len(x)
            
        train_loss = total_train_loss / len(datasets["train"])
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y_flow, y_cong in val_loader:
                x = x.to(device)
                y_flow = y_flow.to(device)
                y_cong = y_cong.to(device)
                
                pred_flow, pred_cong, _ = model(x)
                flow_loss = criterion_flow(pred_flow / 100.0, y_flow / 100.0) * 100.0
                cong_loss = criterion_cong(pred_cong, y_cong)
                loss = flow_loss + 0.4 * cong_loss
                total_val_loss += loss.item() * len(x)
                
        val_loss = total_val_loss / len(datasets["val"])
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            
    print(f"[Train LSTM] Training complete. Best checkpoint saved to {checkpoint_path}")
    plot_loss_curves(history, save_path="reports/figures/lstm_loss.png", title="Traffic Flow LSTM Training & Validation Loss")
    
    # Final Evaluation on Test Set
    print("\n[Train LSTM] Evaluating Best Model on Test Partition...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_true_flows, all_pred_flows = [], []
    all_true_cong, all_pred_cong = [], []
    
    with torch.no_grad():
        for x, y_flow, y_cong in test_loader:
            x = x.to(device)
            pred_flow, pred_cong, _ = model(x)
            
            all_true_flows.append(y_flow.cpu().numpy())
            all_pred_flows.append(pred_flow.cpu().numpy())
            
            preds_cong = torch.argmax(pred_cong, dim=1).cpu().numpy()
            all_true_cong.extend(y_cong.cpu().numpy().tolist())
            all_pred_cong.extend(preds_cong.tolist())
            
    all_true_flows = np.concatenate(all_true_flows, axis=0)
    all_pred_flows = np.concatenate(all_pred_flows, axis=0)
    
    # Regression metrics for each horizon step
    horizon_metrics = {}
    for h in range(forecast_horizon):
        h_min = (h + 1) * 5
        metrics_h = calculate_regression_metrics(all_true_flows[:, h], all_pred_flows[:, h])
        horizon_metrics[f"t+{h_min}min"] = metrics_h
        
    class_names = [CONGESTION_LEVELS[i] for i in range(4)]
    cong_metrics = calculate_classification_metrics(all_true_cong, all_pred_cong, class_names=class_names)
    
    print("\n" + "=" * 55)
    print("LSTM TRAFFIC FLOW EVALUATION RESULTS:")
    print("=" * 55)
    for h_name, m_dict in horizon_metrics.items():
        print(f"\nForecast Horizon {h_name}:")
        for k, v in m_dict.items():
            print(f"  {k}: {v}")
            
    print("\nFuture Congestion State Classification:")
    for k in ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]:
        print(f"  {k}: {cong_metrics[k]}")
    print("\nClassification Report:\n", cong_metrics["Classification Report"])
    
    # Save visual charts
    plot_traffic_flow_prediction(
        all_true_flows, all_pred_flows, horizon_idx=0,
        horizon_label="t + 5 mins", num_points=120,
        save_path="reports/figures/flow_prediction_curve.png"
    )
    
    plot_confusion_matrix(
        cong_metrics["Confusion Matrix"],
        class_names=class_names,
        save_path="reports/figures/lstm_confusion_matrix.png",
        title="LSTM Congestion Classification Confusion Matrix"
    )
    
    summary = {
        "multi_horizon_flow_metrics": horizon_metrics,
        "classification_metrics": {k: v for k, v in cong_metrics.items() if k not in ["Confusion Matrix", "Classification Report"]}
    }
    with open("reports/lstm_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary


if __name__ == "__main__":
    train_lstm(epochs=18, batch_size=32)
