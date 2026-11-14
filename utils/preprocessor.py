"""
preprocessor.py
Data Preprocessing, Augmentation, Normalization, and Dataset Split Utilities
for both Road-Camera Vision Data (CNN) and Sequential Traffic Flow Data (LSTM).
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


# ==========================================
# 1. Vision Preprocessing & Augmentation
# ==========================================

def get_image_transforms(img_size=(224, 224)):
    """Returns training augmentations and validation/testing preprocessing pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class TrafficImageDataset(Dataset):
    """PyTorch Dataset for Road-Camera Frames."""
    def __init__(self, metadata_df, images_dir, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
            
        vehicle_count = torch.tensor(float(row["vehicle_count"]), dtype=torch.float32).unsqueeze(0)
        congestion_level = torch.tensor(int(row["congestion_level"]), dtype=torch.long)
        
        return img, vehicle_count, congestion_level


def split_image_dataset(metadata_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Splits image metadata into train, val, and test subsets."""
    df_shuffled = metadata_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]
    
    return train_df, val_df, test_df


# ==========================================
# 2. Time-Series Preprocessing & Sliding Window
# ==========================================

class TrafficScaler:
    """Standard / MinMax feature scaler with fit and transform capability."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

    def transform(self, data):
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data, feature_idx=0):
        """Inverses scaling for a specific target feature index (default: flow at index 0)."""
        if isinstance(feature_idx, int):
            return data * self.std_[feature_idx] + self.mean_[feature_idx]
        return data * self.std_ + self.mean_


def build_time_features(df):
    """Extracts cyclical trigonometric temporal features and calendar encodings."""
    df_feat = df.copy()
    
    # Hour cyclical encoding (0-24)
    hours = df_feat["hour_of_day"].values
    df_feat["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df_feat["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    
    # Day cyclical encoding (0-6)
    days = df_feat["day_of_week"].values
    df_feat["sin_day"] = np.sin(2 * np.pi * days / 7.0)
    df_feat["cos_day"] = np.cos(2 * np.pi * days / 7.0)
    
    return df_feat


def create_sliding_windows(data, targets_flow, targets_congestion, seq_len=12, forecast_horizon=3):
    """
    Creates temporal sliding sequences for RNN/LSTM.
    Input window: t - seq_len + 1 ... t
    Target window: t + 1 ... t + forecast_horizon
    """
    x_seqs = []
    y_flows = []
    y_congestions = []
    
    total_len = len(data)
    for i in range(total_len - seq_len - forecast_horizon + 1):
        x_window = data[i:i + seq_len]
        y_flow_window = targets_flow[i + seq_len:i + seq_len + forecast_horizon]
        
        # Immediate future congestion level (at t + 1)
        y_cong = targets_congestion[i + seq_len]
        
        x_seqs.append(x_window)
        y_flows.append(y_flow_window)
        y_congestions.append(y_cong)
        
    return np.array(x_seqs, dtype=np.float32), np.array(y_flows, dtype=np.float32), np.array(y_congestions, dtype=np.int64)


class TrafficSequenceDataset(Dataset):
    """PyTorch Dataset for Sequential Traffic Prediction."""
    def __init__(self, x_seqs, y_flows, y_congestions):
        self.x = torch.from_numpy(x_seqs)
        self.y_flow = torch.from_numpy(y_flows)
        self.y_cong = torch.from_numpy(y_congestions)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_flow[idx], self.y_cong[idx]


def prepare_timeseries_data(csv_path, seq_len=12, forecast_horizon=3, train_ratio=0.70, val_ratio=0.15):
    """
    Complete temporal pipeline: loads CSV, generates cyclic features,
    scales numerical features without future leakage, generates windows, and splits.
    """
    raw_df = pd.read_csv(csv_path)
    feat_df = build_time_features(raw_df)
    
    feature_cols = [
        "vehicle_flow", "avg_speed_kmh", "road_occupancy_pct",
        "sin_hour", "cos_hour", "sin_day", "cos_day"
    ]
    
    raw_features = feat_df[feature_cols].values
    flows = feat_df["vehicle_flow"].values
    congestions = feat_df["congestion_level"].values
    
    n_samples = len(raw_features)
    train_idx = int(n_samples * train_ratio)
    val_idx = int(n_samples * (train_ratio + val_ratio))
    
    # Fit scaler solely on training partition to prevent lookahead bias
    scaler = TrafficScaler()
    scaler.fit(raw_features[:train_idx])
    
    scaled_features = scaler.transform(raw_features)
    
    # Create windows for entire dataset
    x_all, y_flow_all, y_cong_all = create_sliding_windows(
        scaled_features, flows, congestions, seq_len=seq_len, forecast_horizon=forecast_horizon
    )
    
    # Split windows chronologically
    train_end = int(len(x_all) * train_ratio)
    val_end = int(len(x_all) * (train_ratio + val_ratio))
    
    x_train, y_f_train, y_c_train = x_all[:train_end], y_flow_all[:train_end], y_cong_all[:train_end]
    x_val, y_f_val, y_c_val = x_all[train_end:val_end], y_flow_all[train_end:val_end], y_cong_all[train_end:val_end]
    x_test, y_f_test, y_c_test = x_all[val_end:], y_flow_all[val_end:], y_cong_all[val_end:]
    
    datasets = {
        "train": TrafficSequenceDataset(x_train, y_f_train, y_c_train),
        "val": TrafficSequenceDataset(x_val, y_f_val, y_c_val),
        "test": TrafficSequenceDataset(x_test, y_f_test, y_c_test)
    }
    
    return datasets, scaler, feature_cols
