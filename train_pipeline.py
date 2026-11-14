"""
train_pipeline.py
Master Execution Script for the Entire CNN-LSTM Traffic Surveillance & Forecasting Project.
Runs dataset synthesis, CNN training/evaluation, LSTM training/evaluation,
and generates the real-time HUD demonstration frame.
"""

import os
import sys
import cv2
import numpy as np

from train_cnn import train_cnn
from train_lstm import train_lstm
from models.cnn_lstm_pipeline import TrafficPipeline
from utils.visualizer import render_demonstration_frame
from utils.dataset_generator import create_road_background, render_vehicle


def run_full_pipeline():
    print("=" * 65)
    print("TRAFFIC FLOW PREDICTION PROJECT: END-TO-END PIPELINE RUNNER")
    print("=" * 65)
    
    # Stage 1-5: Train CNN
    print("\n>>> STAGE 1: Training CNN Vehicle Detection & Congestion Model...")
    cnn_results = train_cnn(epochs=12, batch_size=16)
    
    # Stage 1-5: Train LSTM
    print("\n>>> STAGE 2: Training LSTM Temporal Flow Forecaster...")
    lstm_results = train_lstm(epochs=16, batch_size=32)
    
    # Stage 10: Generate Demonstration Frame & Video Preview
    print("\n>>> STAGE 3: Running Integrated Demonstration Simulation...")
    pipeline = TrafficPipeline(
        cnn_model_path="models/checkpoints/traffic_cnn.pt",
        lstm_model_path="models/checkpoints/traffic_lstm.pt"
    )
    
    # Create a realistic sample camera frame with vehicles
    demo_bg = create_road_background(640, 360)
    # Render several vehicles simulating peak hour traffic
    vehicles = [
        (260, 160, 0.6, 0),   # Car
        (380, 180, 0.7, 1),   # SUV
        (180, 240, 0.9, 0),   # Car
        (460, 260, 1.0, 2),   # Bus/Truck
        (320, 290, 1.1, 0),   # Car
        (230, 310, 1.2, 3),   # Motorcycle
        (410, 320, 1.2, 1),   # SUV
    ]
    for x, y, scale, cid in vehicles:
        render_vehicle(demo_bg, x, y, scale, cid)
        
    detected_count, c_class, c_label, latent = pipeline.process_frame(demo_bg)
    
    # Mock recent history for LSTM
    mock_history = np.zeros((12, 7), dtype=np.float32)
    # Populate historical trends (e.g. rising flow before morning peak)
    for i in range(12):
        base_f = 45.0 + i * 4.5
        mock_history[i] = [base_f, 75.0 - i * 2.0, 30.0 + i * 2.5, 0.7, 0.2, 0.0, 0.0]
        
    predicted_flows, pred_congestion = pipeline.forecast_future_flow(mock_history)
    
    demo_overlay = render_demonstration_frame(
        demo_bg,
        current_count=detected_count,
        predicted_flows=predicted_flows,
        congestion_state=pred_congestion,
        save_path="reports/figures/demo_frame.png"
    )
    
    print("\n" + "=" * 65)
    print("PROJECT PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 65)
    print(f"Sample Frame Ingestion:")
    print(f"  Live CNN Detected Vehicle Count: {detected_count:.1f}")
    print(f"  Live CNN Congestion Estimate:    {c_label}")
    print(f"LSTM Short-Term Flow Forecast:")
    print(f"  +5  min: {predicted_flows[0]:.1f} vehicles")
    print(f"  +10 min: {predicted_flows[1]:.1f} vehicles")
    print(f"  +15 min: {predicted_flows[2]:.1f} vehicles")
    print(f"  Predicted Congestion State:      {pred_congestion}")
    print(f"Demonstration HUD saved to: reports/figures/demo_frame.png")
    print("=" * 65)


if __name__ == "__main__":
    run_full_pipeline()
