"""
visualizer.py
Visualization Utilities for Traffic Flow Prediction and Vehicle Detection.
Generates training history curves, confusion matrix plots, actual vs. predicted flow charts,
and road-camera frame detection overlays.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_loss_curves(history, save_path="reports/figures/loss_curves.png", title="Training and Validation Loss"):
    """Plots training and validation loss over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
        
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Loss curve saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path="reports/figures/confusion_matrix.png", title="Confusion Matrix"):
    """Plots and saves a formatted confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Ground Truth Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    
    # Overlay count text
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )
            
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Confusion matrix saved to {save_path}")


def plot_traffic_flow_prediction(y_true, y_pred, horizon_idx=0, horizon_label="t + 1 (5 mins)",
                                 num_points=120, save_path="reports/figures/flow_prediction_curve.png"):
    """
    Plots ground truth vs. predicted vehicle flow across consecutive time intervals.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    true_slice = y_true[:num_points, horizon_idx] if y_true.ndim > 1 else y_true[:num_points]
    pred_slice = y_pred[:num_points, horizon_idx] if y_pred.ndim > 1 else y_pred[:num_points]
    
    time_steps = range(len(true_slice))
    
    plt.plot(time_steps, true_slice, color="#1f77b4", label="Actual Traffic Flow", linewidth=2.2)
    plt.plot(time_steps, pred_slice, color="#d62728", linestyle="--", label=f"Predicted Flow ({horizon_label})", linewidth=2.0)
    
    # Highlight residual error band
    plt.fill_between(time_steps, true_slice, pred_slice, color="gray", alpha=0.2, label="Prediction Residual Error")
    
    plt.title(f"Short-Term Traffic Flow Forecasting vs. Ground Truth ({horizon_label})", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step (5-Minute Intervals)", fontsize=12)
    plt.ylabel("Vehicle Flow (Vehicles / 5-min)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Traffic flow prediction plot saved to {save_path}")


def render_demonstration_frame(frame, current_count, predicted_flows, congestion_state, save_path="reports/figures/demo_frame.png"):
    """
    Renders an informative HUD overlay on top of road camera video frame.
    Shows current detected count, predicted future flows, and congestion alert status.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Dashboard Header banner (semi-transparent dark)
    header_h = 75
    hud_bg = overlay.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, header_h), (25, 25, 30), -1)
    cv2.addWeighted(hud_bg, 0.85, overlay, 0.15, 0, overlay)
    
    # Titles & Statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, "ROAD CAMERA REAL-TIME TRAFFIC MONITORING", (15, 25), font, 0.65, (255, 255, 255), 2)
    
    count_str = f"Live Vehicle Count: {int(current_count)}"
    cv2.putText(overlay, count_str, (15, 55), font, 0.6, (50, 220, 255), 2)
    
    # Congestion Status Badge color
    # Free: Green, Moderate: Yellow, Heavy: Orange, Severe: Red
    badge_colors = {
        "Free Flow": (50, 200, 50),
        "Moderate Flow": (50, 215, 255),
        "Heavy Congestion": (30, 130, 255),
        "Severe Gridlock": (40, 40, 240)
    }
    badge_color = badge_colors.get(congestion_state, (200, 200, 200))
    
    state_str = f"Status: {congestion_state.upper()}"
    cv2.rectangle(overlay, (w - 240, 12), (w - 15, 45), badge_color, -1)
    cv2.putText(overlay, state_str, (w - 230, 34), font, 0.5, (0, 0, 0), 2)
    
    # Forecast Drawer (Bottom Banner)
    footer_h = 65
    footer_y = h - footer_h
    hud_footer = overlay.copy()
    cv2.rectangle(hud_footer, (0, footer_y), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(hud_footer, 0.85, overlay, 0.15, 0, overlay)
    
    cv2.putText(overlay, "LSTM SHORT-TERM FORECAST:", (15, footer_y + 25), font, 0.55, (200, 200, 200), 1)
    
    f1, f2, f3 = predicted_flows[0], predicted_flows[1], predicted_flows[2]
    forecast_text = f"+5 min: {f1:.1f} veh/int | +10 min: {f2:.1f} veh/int | +15 min: {f3:.1f} veh/int"
    cv2.putText(overlay, forecast_text, (15, footer_y + 50), font, 0.6, (100, 255, 120), 2)
    
    cv2.imwrite(save_path, overlay)
    print(f"[Visualizer] Demonstration frame overlay saved to {save_path}")
    return overlay
