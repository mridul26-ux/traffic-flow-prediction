"""
cnn_lstm_pipeline.py
End-to-End Integrated CNN-LSTM Traffic Surveillance & Forecasting Pipeline.
Orchestrates live road-camera frame ingestion, CNN spatial vehicle detection/feature extraction,
temporal sequence buffering, and LSTM short-term traffic flow prediction.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_detector import TrafficCNN
from models.lstm_predictor import TrafficLSTM
from utils.dataset_generator import CONGESTION_LEVELS


class TrafficPipeline:
    """
    End-to-End Deep Learning Pipeline for Traffic Flow Prediction:
    Camera Frames -> CNN Spatial Density / Features -> Sequence Buffer -> LSTM Temporal Predictor
    """
    def __init__(self, cnn_model_path=None, lstm_model_path=None, seq_len=12, forecast_horizon=3, device="cpu"):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        # Initialize CNN
        self.cnn = TrafficCNN(feature_dim=64, num_classes=4).to(self.device)
        if cnn_model_path and os.path.exists(cnn_model_path):
            state_dict = torch.load(cnn_model_path, map_location=self.device)
            self.cnn.load_state_dict(state_dict)
            print(f"[Pipeline] Loaded CNN weights from {cnn_model_path}")
        self.cnn.eval()
        
        # Initialize LSTM
        self.lstm = TrafficLSTM(input_dim=7, hidden_dim=64, forecast_horizon=forecast_horizon).to(self.device)
        if lstm_model_path and os.path.exists(lstm_model_path):
            state_dict = torch.load(lstm_model_path, map_location=self.device)
            self.lstm.load_state_dict(state_dict)
            print(f"[Pipeline] Loaded LSTM weights from {lstm_model_path}")
        self.lstm.eval()
        
        # Frame transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Rolling temporal buffer (holds past seq_len feature vectors)
        self.temporal_buffer = []

    def process_frame(self, frame_bgr):
        """
        Passes a single camera frame through the CNN.
        Returns:
          detected_count: float
          congestion_class: int
          congestion_label: str
          feature_vector: np.ndarray (64-dim)
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_count, pred_logits, latent = self.cnn(tensor)
            
            count_val = max(0.0, float(pred_count.item()))
            c_class = int(torch.argmax(pred_logits, dim=1).item())
            c_label = CONGESTION_LEVELS.get(c_class, "Unknown")
            latent_np = latent.cpu().squeeze(0).numpy()
            
        return count_val, c_class, c_label, latent_np

    def update_buffer(self, feature_row):
        """Appends latest step feature vector to rolling sequence buffer."""
        self.temporal_buffer.append(feature_row)
        if len(self.temporal_buffer) > self.seq_len:
            self.temporal_buffer.pop(0)

    def forecast_future_flow(self, current_sequence=None):
        """
        Predicts future traffic flow using the LSTM over current buffer or input sequence.
        Returns:
          future_flows: list of floats for upcoming intervals [t+1, t+2, t+3]
          predicted_congestion: str
        """
        if current_sequence is None:
            if len(self.temporal_buffer) < self.seq_len:
                # Pad buffer if insufficient history
                pad_needed = self.seq_len - len(self.temporal_buffer)
                first_elem = self.temporal_buffer[0] if self.temporal_buffer else np.zeros(7, dtype=np.float32)
                seq_data = [first_elem] * pad_needed + self.temporal_buffer
            else:
                seq_data = self.temporal_buffer
            x_arr = np.array(seq_data, dtype=np.float32)[np.newaxis, ...]
        else:
            x_arr = np.array(current_sequence, dtype=np.float32)
            if x_arr.ndim == 2:
                x_arr = x_arr[np.newaxis, ...]
                
        tensor_x = torch.from_numpy(x_arr).to(self.device)
        
        with torch.no_grad():
            pred_flow, pred_cong, _ = self.lstm(tensor_x)
            flows = pred_flow.cpu().squeeze(0).numpy().tolist()
            c_idx = int(torch.argmax(pred_cong, dim=1).item())
            c_label = CONGESTION_LEVELS.get(c_idx, "Unknown")
            
        return flows, c_label
