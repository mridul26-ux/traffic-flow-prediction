"""
demo.py
Interactive Traffic Flow Prediction Demonstration.
Simulates a live road-camera video feed, detects vehicles per frame using the CNN,
buffers temporal counts and features, and produces short-term LSTM traffic flow forecasts.
"""

import os
import time
import cv2
import numpy as np

from models.cnn_lstm_pipeline import TrafficPipeline
from utils.dataset_generator import create_road_background, render_vehicle
from utils.visualizer import render_demonstration_frame


def run_interactive_simulation(num_frames=30, output_video_path="reports/figures/traffic_simulation.mp4"):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    cnn_ckpt = "models/checkpoints/traffic_cnn.pt"
    lstm_ckpt = "models/checkpoints/traffic_lstm.pt"
    
    if not os.path.exists(cnn_ckpt) or not os.path.exists(lstm_ckpt):
        print("[Demo Error] Checkpoints not found. Please run train_pipeline.py first!")
        return
        
    pipeline = TrafficPipeline(cnn_model_path=cnn_ckpt, lstm_model_path=lstm_ckpt)
    
    print("=" * 65)
    print("STARTING REAL-TIME TRAFFIC CAMERA SURVEILLANCE & PREDICTION DEMO")
    print("=" * 65)
    print("Simulating live camera feed with moving traffic...")
    
    width, height = 640, 360
    base_bg = create_road_background(width, height)
    
    # Initialize animated vehicle instances: [x, depth_ratio, speed, class_id]
    num_sim_vehicles = 14
    vehicles = []
    for _ in range(num_sim_vehicles):
        depth = np.random.uniform(0.1, 0.95)
        cid = int(np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.12, 0.08]))
        speed = np.random.uniform(0.015, 0.035)
        lane_choice = np.random.choice([-0.25, 0.0, 0.25])
        vehicles.append({"depth": depth, "speed": speed, "cid": cid, "lane": lane_choice})
        
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 5.0, (width, height))
    
    rolling_features = []
    
    for f_idx in range(num_frames):
        frame = base_bg.copy()
        horizon = int(height * 0.32)
        
        # Render vehicles and update positions
        for v in vehicles:
            v["depth"] += v["speed"]
            if v["depth"] > 1.0:
                v["depth"] = 0.05
                v["cid"] = int(np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.12, 0.08]))
                v["lane"] = np.random.choice([-0.25, 0.0, 0.25])
                
            depth = v["depth"]
            scale = 0.4 + 1.1 * depth
            y = int(horizon + depth * (height - horizon))
            
            # Center road position at this depth
            road_center = width * 0.5
            road_w = (width * 0.30) + depth * (width * 0.60)
            x = int(road_center + v["lane"] * road_w)
            
            render_vehicle(frame, x, y, scale, v["cid"])
            
        # Ingest into CNN
        detected_count, c_class, c_label, latent = pipeline.process_frame(frame)
        
        # Simulated continuous traffic context
        speed_est = max(15.0, 85.0 - (detected_count * 3.2))
        occ_est = min(95.0, detected_count * 4.5)
        feature_row = [detected_count * 8.0, speed_est, occ_est, 0.7, -0.3, 0.0, 0.0]
        rolling_features.append(feature_row)
        if len(rolling_features) > 12:
            rolling_features.pop(0)
            
        predicted_flows, pred_congestion = pipeline.forecast_future_flow(rolling_features)
        
        # Render HUD
        hud_frame = render_demonstration_frame(
            frame,
            current_count=detected_count,
            predicted_flows=predicted_flows,
            congestion_state=pred_congestion,
            save_path="reports/figures/demo_frame.png"
        )
        
        if video_writer is not None:
            video_writer.write(hud_frame)
            
        print(f"Frame {f_idx + 1:02d}/{num_frames:02d} | CNN Vehicles: {detected_count:.1f} | Live: {c_label} | Forecast (+5m): {predicted_flows[0]:.1f} veh")
        
    if video_writer is not None:
        video_writer.release()
        print(f"\n[Demo] Simulation video rendered and saved to: {output_video_path}")
        
    print("[Demo] Interactive demonstration finished successfully!")


if __name__ == "__main__":
    run_interactive_simulation(num_frames=20)
