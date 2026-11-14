"""
dataset_generator.py
Synthetic Road-Camera Footage & Traffic Time-Series Dataset Generator.
Provides realistic synthetic road-camera video frames and time-series traffic flow data
with diurnal patterns (rush hours, off-peak periods, and sudden congestion shocks).
"""

import os
import math
import random
import numpy as np
import pandas as pd
import cv2
from datetime import datetime, timedelta

# Class mappings
VEHICLE_CLASSES = {
    0: {"name": "Car", "color": (50, 150, 255), "size": (35, 20)},        # Blue-ish
    1: {"name": "SUV", "color": (50, 200, 100), "size": (45, 24)},        # Green-ish
    2: {"name": "Truck/Bus", "color": (220, 100, 50), "size": (70, 28)},  # Orange-red
    3: {"name": "Motorcycle", "color": (200, 50, 200), "size": (20, 12)}  # Magenta
}

CONGESTION_LEVELS = {
    0: "Free Flow",
    1: "Moderate Flow",
    2: "Heavy Congestion",
    3: "Severe Gridlock"
}


def create_road_background(width=640, height=360):
    """Generates a realistic road perspective background."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 110  # Scenery gray/green
    
    # Sky / Horizon top 25%
    horizon = int(height * 0.30)
    frame[:horizon, :] = [180, 160, 130]  # Light sky
    
    # Road polygon (perspective: narrow at horizon, wide at bottom)
    road_top_left = (int(width * 0.35), horizon)
    road_top_right = (int(width * 0.65), horizon)
    road_bottom_left = (int(width * 0.05), height)
    road_bottom_right = (int(width * 0.95), height)
    
    road_poly = np.array([road_top_left, road_top_right, road_bottom_right, road_bottom_left], dtype=np.int32)
    cv2.fillPoly(frame, [road_poly], (55, 55, 58))  # Dark asphalt
    
    # Road shoulders
    shoulder_left = np.array([(int(width * 0.32), horizon), road_top_left, road_bottom_left, (0, height)], dtype=np.int32)
    shoulder_right = np.array([road_top_right, (int(width * 0.68), horizon), (width, height), road_bottom_right], dtype=np.int32)
    cv2.fillPoly(frame, [shoulder_left], (70, 95, 70))
    cv2.fillPoly(frame, [shoulder_right], (70, 95, 70))
    
    # Lane divider lines (3 lanes: left, center, right)
    for t in [0.33, 0.67]:
        pt1 = (int(road_top_left[0] + t * (road_top_right[0] - road_top_left[0])), horizon)
        pt2 = (int(road_bottom_left[0] + t * (road_bottom_right[0] - road_bottom_left[0])), height)
        # Dashed line
        num_dashes = 12
        for i in range(num_dashes):
            if i % 2 == 0:
                s = i / float(num_dashes)
                e = (i + 0.6) / float(num_dashes)
                p_start = (int(pt1[0] + s * (pt2[0] - pt1[0])), int(pt1[1] + s * (pt2[1] - pt1[1])))
                p_end = (int(pt1[0] + e * (pt2[0] - pt1[0])), int(pt1[1] + e * (pt2[1] - pt1[1])))
                thickness = max(1, int(1 + 3 * e))
                cv2.line(frame, p_start, p_end, (220, 220, 220), thickness)
                
    return frame


def render_vehicle(frame, x, y, scale, class_id):
    """Draws a vehicle with perspective scaling and bounding box coordinates."""
    v_info = VEHICLE_CLASSES[class_id]
    base_w, base_h = v_info["size"]
    w = max(8, int(base_w * scale))
    h = max(6, int(base_h * scale))
    color = v_info["color"]
    
    x1 = int(x - w // 2)
    y1 = int(y - h // 2)
    x2 = x1 + w
    y2 = y1 + h
    
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(w_frame - 1, x1))
    y1 = max(0, min(h_frame - 1, y1))
    x2 = max(1, min(w_frame, x2))
    y2 = max(1, min(h_frame, y2))
    
    # Vehicle shadow
    cv2.rectangle(frame, (x1, y2 - 2), (x2, y2 + 2), (30, 30, 30), -1)
    # Vehicle body
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # Windshield / Roof highlight
    roof_w = max(2, int((x2 - x1) * 0.5))
    roof_h = max(2, int((y2 - y1) * 0.4))
    rx1 = x1 + (x2 - x1 - roof_w) // 2
    ry1 = y1 + (y2 - y1 - roof_h) // 2
    cv2.rectangle(frame, (rx1, ry1), (rx1 + roof_w, ry1 + roof_h), (30, 30, 40), -1)
    
    # Headlights or taillights
    cv2.circle(frame, (x1 + 3, y2 - 3), max(1, int(2 * scale)), (0, 0, 220), -1)
    cv2.circle(frame, (x2 - 3, y2 - 3), max(1, int(2 * scale)), (0, 0, 220), -1)
    
    return [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]


def generate_camera_dataset(output_dir, num_samples=300, img_size=(224, 224)):
    """
    Generates annotated road camera frames categorized by vehicle density/count.
    Saves image frames and an annotations CSV.
    """
    images_dir = os.path.join(output_dir, "frames")
    os.makedirs(images_dir, exist_ok=True)
    
    records = []
    base_bg = create_road_background(640, 360)
    
    for idx in range(num_samples):
        frame = base_bg.copy()
        h_frame, w_frame = frame.shape[:2]
        
        # Determine congestion regime for this sample
        # 0: Low (1-4 vehicles), 1: Moderate (5-10), 2: Heavy (11-18), 3: Extreme (19-25)
        regime = np.random.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15])
        if regime == 0:
            num_vehicles = random.randint(1, 4)
        elif regime == 1:
            num_vehicles = random.randint(5, 10)
        elif regime == 2:
            num_vehicles = random.randint(11, 18)
        else:
            num_vehicles = random.randint(19, 26)
            
        boxes = []
        counts_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
        
        horizon = int(h_frame * 0.32)
        
        # Populate lanes with vehicles
        for _ in range(num_vehicles):
            # Pick a vertical depth y (from near horizon to bottom)
            depth_ratio = random.uniform(0.1, 0.95)
            y = int(horizon + depth_ratio * (h_frame - horizon))
            scale = 0.4 + 1.1 * depth_ratio
            
            # Road width at this y
            left_x = int(w_frame * 0.35 + depth_ratio * (w_frame * 0.05 - w_frame * 0.35))
            right_x = int(w_frame * 0.65 + depth_ratio * (w_frame * 0.95 - w_frame * 0.65))
            
            x = random.randint(left_x + 10, right_x - 10)
            
            # Select vehicle class with realistic highway proportions
            # 60% Car, 20% SUV, 12% Truck/Bus, 8% Motorcycle
            class_id = int(np.random.choice([0, 1, 2, 3], p=[0.60, 0.20, 0.12, 0.08]))
            counts_by_class[class_id] += 1
            
            bbox = render_vehicle(frame, x, y, scale, class_id)
            boxes.append((class_id, bbox))
            
        # Add realistic camera noise / illumination variance
        brightness = random.uniform(0.75, 1.25)
        frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        # Save image
        img_name = f"frame_{idx:05d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        
        # Resize to requested network input size
        resized_frame = cv2.resize(frame, img_size)
        cv2.imwrite(img_path, resized_frame)
        
        # Classify overall congestion level based on count
        if num_vehicles <= 4:
            c_level = 0  # Free Flow
        elif num_vehicles <= 10:
            c_level = 1  # Moderate
        elif num_vehicles <= 18:
            c_level = 2  # Heavy
        else:
            c_level = 3  # Severe
            
        records.append({
            "image_name": img_name,
            "vehicle_count": num_vehicles,
            "car_count": counts_by_class[0],
            "suv_count": counts_by_class[1],
            "truck_count": counts_by_class[2],
            "motorcycle_count": counts_by_class[3],
            "congestion_level": c_level,
            "congestion_label": CONGESTION_LEVELS[c_level]
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "camera_dataset_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Dataset Generator] Generated {num_samples} road-camera frames in {images_dir}")
    print(f"[Dataset Generator] Saved metadata to {csv_path}")
    return df


def generate_traffic_timeseries(num_days=30, interval_minutes=5, output_csv="data/raw/traffic_timeseries.csv"):
    """
    Generates a realistic traffic flow time-series over multiple days.
    Includes:
      - Diurnal curve (morning rush hour 7:30-9:30, evening rush 17:00-19:30)
      - Weekday vs. Weekend variations
      - Stochastic bursts and sudden traffic incidents
      - Correlated metrics: flow (vehicles/5-min), average speed (km/h), road occupancy (%)
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    total_steps = int((num_days * 24 * 60) / interval_minutes)
    
    timestamps = []
    vehicle_counts = []
    speeds = []
    occupancies = []
    congestion_levels = []
    weather_factors = []
    
    # Pre-select some random incident times (e.g. accidents or stalled vehicles)
    incident_steps = set(random.sample(range(50, total_steps - 50), k=int(num_days * 1.5)))
    incident_durations = {step: random.randint(6, 18) for step in incident_steps}  # 30-90 min duration
    
    active_incident_decay = 0.0
    
    for step in range(total_steps):
        current_time = start_time + timedelta(minutes=step * interval_minutes)
        timestamps.append(current_time)
        
        hour = current_time.hour + current_time.minute / 60.0
        is_weekend = current_time.weekday() >= 5
        
        # Base daily cyclical pattern
        # Morning peak around 8:30 AM, Evening peak around 18:00 (6 PM)
        morning_peak = math.exp(-((hour - 8.5) ** 2) / 3.0)
        evening_peak = math.exp(-((hour - 18.0) ** 2) / 4.0)
        night_trough = 0.15 + 0.1 * math.cos((hour - 3.0) * math.pi / 12.0)
        
        if is_weekend:
            # Weekend peak is broader in early afternoon
            weekend_midday = math.exp(-((hour - 14.0) ** 2) / 8.0)
            base_pattern = 0.2 + 0.6 * weekend_midday
        else:
            base_pattern = 0.2 + 0.7 * morning_peak + 0.85 * evening_peak
            
        base_pattern = max(0.08, min(1.0, base_pattern))
        
        # Check for incident start
        if step in incident_steps:
            active_incident_decay = 1.0
            
        # Incident causes sudden speed drop and vehicle queueing
        incident_impact = active_incident_decay
        if active_incident_decay > 0:
            active_incident_decay = max(0.0, active_incident_decay - 0.08)
            
        # Random fluctuation (weather, random groups)
        noise = np.random.normal(0, 0.05)
        
        # Expected traffic flow (vehicles per 5 minutes)
        # Max highway capacity per 5 min ~ 150 vehicles across 3 lanes
        capacity = 140.0
        expected_flow = capacity * (base_pattern + noise)
        
        if incident_impact > 0:
            # Stalled flow: vehicle flow drops, congestion spikes
            expected_flow *= (1.0 - 0.45 * incident_impact)
            
        flow = max(5, int(expected_flow + np.random.normal(0, 4)))
        
        # Fundamental traffic relationship: Flow = Density * Speed (Greenberg / Greenshields model)
        # At high flow / density, speed drops
        density_ratio = min(1.0, (flow / capacity) + 0.3 * incident_impact)
        free_flow_speed = 90.0  # km/h
        min_congested_speed = 12.0
        
        speed = free_flow_speed * (1.0 - (density_ratio ** 1.8)) + np.random.normal(0, 2.5)
        speed = max(min_congested_speed, min(free_flow_speed + 5, speed))
        
        occupancy = min(99.0, max(2.0, (density_ratio * 75.0) + np.random.normal(0, 1.5)))
        
        # Categorize congestion state
        if occupancy < 25.0 and speed > 70.0:
            c_state = 0  # Free Flow
        elif occupancy < 48.0 and speed > 45.0:
            c_state = 1  # Moderate Flow
        elif occupancy < 75.0 or speed > 22.0:
            c_state = 2  # Heavy Congestion
        else:
            c_state = 3  # Severe Gridlock
            
        vehicle_counts.append(flow)
        speeds.append(round(float(speed), 2))
        occupancies.append(round(float(occupancy), 2))
        congestion_levels.append(c_state)
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "vehicle_flow": vehicle_counts,
        "avg_speed_kmh": speeds,
        "road_occupancy_pct": occupancies,
        "congestion_level": congestion_levels,
        "is_weekend": [1 if t.weekday() >= 5 else 0 for t in timestamps],
        "hour_of_day": [t.hour + t.minute / 60.0 for t in timestamps],
        "day_of_week": [t.weekday() for t in timestamps]
    })
    
    df.to_csv(output_csv, index=False)
    print(f"[Dataset Generator] Generated {len(df)} time-series steps ({num_days} days) saved to {output_csv}")
    return df


if __name__ == "__main__":
    out_dir = "data/raw"
    generate_camera_dataset(out_dir, num_samples=150)
    generate_traffic_timeseries(num_days=14, output_csv=os.path.join(out_dir, "traffic_timeseries.csv"))
