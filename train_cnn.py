"""
train_cnn.py
Training and Evaluation Script for Road-Camera CNN Vehicle Detection & Congestion Model.
Trains on road-camera images, evaluates performance with multi-metric suite,
saves model checkpoint and visualization plots.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from models.cnn_detector import TrafficCNN
from utils.dataset_generator import generate_camera_dataset, CONGESTION_LEVELS
from utils.preprocessor import (
    get_image_transforms, TrafficImageDataset, split_image_dataset
)
from utils.metrics import calculate_regression_metrics, calculate_classification_metrics
from utils.visualizer import plot_loss_curves, plot_confusion_matrix


def train_cnn(data_dir="data/raw", epochs=15, batch_size=16, lr=1e-3, device="cpu"):
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    device = torch.device(device)
    print(f"[Train CNN] Using device: {device}")
    
    # Check if dataset exists, otherwise generate
    csv_path = os.path.join(data_dir, "camera_dataset_metadata.csv")
    images_dir = os.path.join(data_dir, "frames")
    
    if not os.path.exists(csv_path) or not os.path.exists(images_dir):
        print("[Train CNN] Dataset not found. Generating synthetic road-camera dataset...")
        df = generate_camera_dataset(data_dir, num_samples=250)
    else:
        df = pd.read_csv(csv_path)
        
    print(f"[Train CNN] Loaded metadata with {len(df)} images.")
    
    # Split
    train_df, val_df, test_df = split_image_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    print(f"[Train CNN] Dataset Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # DataLoaders
    train_transform, val_transform = get_image_transforms(img_size=(224, 224))
    train_loader = DataLoader(TrafficImageDataset(train_df, images_dir, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrafficImageDataset(val_df, images_dir, val_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TrafficImageDataset(test_df, images_dir, val_transform), batch_size=batch_size, shuffle=False)
    
    # Model, Losses, Optimizer
    model = TrafficCNN(feature_dim=64, num_classes=4).to(device)
    criterion_count = nn.SmoothL1Loss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    checkpoint_path = "models/checkpoints/traffic_cnn.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for images, counts, labels in train_loader:
            images = images.to(device)
            counts = counts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            pred_count, pred_logits, _ = model(images)
            
            loss_c = criterion_count(pred_count, counts)
            loss_cls = criterion_class(pred_logits, labels)
            loss = loss_c + 0.6 * loss_cls
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(images)
            
        train_loss = total_train_loss / len(train_df)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, counts, labels in val_loader:
                images = images.to(device)
                counts = counts.to(device)
                labels = labels.to(device)
                
                pred_count, pred_logits, _ = model(images)
                loss_c = criterion_count(pred_count, counts)
                loss_cls = criterion_class(pred_logits, labels)
                loss = loss_c + 0.6 * loss_cls
                total_val_loss += loss.item() * len(images)
                
        val_loss = total_val_loss / len(val_df)
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            
    print(f"[Train CNN] Training complete. Best checkpoint saved to {checkpoint_path}")
    plot_loss_curves(history, save_path="reports/figures/cnn_loss.png", title="CNN Training & Validation Loss")
    
    # Final Test Set Evaluation
    print("\n[Train CNN] Evaluating Best Model on Held-out Test Set...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    y_true_counts, y_pred_counts = [], []
    y_true_cls, y_pred_cls = [], []
    
    with torch.no_grad():
        for images, counts, labels in test_loader:
            images = images.to(device)
            pred_count, pred_logits, _ = model(images)
            
            y_true_counts.extend(counts.squeeze(1).cpu().numpy().tolist())
            y_pred_counts.extend(pred_count.squeeze(1).cpu().numpy().tolist())
            
            preds = torch.argmax(pred_logits, dim=1).cpu().numpy().tolist()
            y_true_cls.extend(labels.cpu().numpy().tolist())
            y_pred_cls.extend(preds)
            
    # Compute Metrics
    count_metrics = calculate_regression_metrics(y_true_counts, y_pred_counts)
    class_names = [CONGESTION_LEVELS[i] for i in range(4)]
    cls_metrics = calculate_classification_metrics(y_true_cls, y_pred_cls, class_names=class_names)
    
    print("\n" + "=" * 50)
    print("CNN TEST EVALUATION RESULTS:")
    print("=" * 50)
    print("Vehicle Count Density Estimation:")
    for k, v in count_metrics.items():
        print(f"  {k}: {v}")
    print("\nTraffic Congestion Classification:")
    for k in ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]:
        print(f"  {k}: {cls_metrics[k]}")
    print("\nClassification Report:\n", cls_metrics["Classification Report"])
    
    # Save confusion matrix plot
    plot_confusion_matrix(
        cls_metrics["Confusion Matrix"],
        class_names=class_names,
        save_path="reports/figures/cnn_confusion_matrix.png",
        title="CNN Congestion Classification Confusion Matrix"
    )
    
    # Save metrics JSON
    summary = {
        "vehicle_count_metrics": count_metrics,
        "classification_metrics": {k: v for k, v in cls_metrics.items() if k not in ["Confusion Matrix", "Classification Report"]}
    }
    with open("reports/cnn_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary


if __name__ == "__main__":
    train_cnn(epochs=12, batch_size=16)
