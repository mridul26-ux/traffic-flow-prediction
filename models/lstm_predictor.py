"""
lstm_predictor.py
Recurrent Neural Network (LSTM / GRU) for Short-Term Traffic Flow Prediction.
Takes historical sequences of vehicle counts, traffic speed, road occupancy, and temporal features,
and predicts future traffic flow volume and congestion levels across short-term prediction horizons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Calculates temporal attention weights over LSTM hidden states across the sequence."""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class TrafficLSTM(nn.Module):
    """
    Traffic Flow Forecasting Model combining bidirectional LSTM with Temporal Attention.
    Architecture:
      - Linear Input Feature Embedding
      - 2-layer Bidirectional LSTM
      - Temporal Attention Context Layer
      - Dual Heads:
          1. Multi-Step Flow Regressor (predicts next H steps, e.g. H=3 steps: 5m, 10m, 15m)
          2. Future Congestion Level Classifier
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, forecast_horizon=3, num_classes=4, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Bidirectional outputs 2 * hidden_dim
        self.attention = TemporalAttention(hidden_dim * 2)
        
        # Dense feature aggregator
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head 1: Multi-step continuous flow prediction (vehicles per interval)
        self.flow_head = nn.Linear(hidden_dim, forecast_horizon)
        
        # Head 2: Future traffic congestion classification
        self.congestion_head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
          x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
          pred_flow: Tensor of shape (batch_size, forecast_horizon)
          pred_congestion: Tensor of shape (batch_size, num_classes)
          attn_weights: Tensor of shape (batch_size, seq_len, 1)
        """
        # Linear projection
        proj = self.input_proj(x)
        
        # LSTM sequence processing
        lstm_out, _ = self.lstm(proj)  # (batch, seq_len, 2 * hidden_dim)
        
        # Attention aggregation
        context, attn_weights = self.attention(lstm_out)  # (batch, 2 * hidden_dim)
        
        feat = self.fc(context)  # (batch, hidden_dim)
        
        pred_flow = self.flow_head(feat)
        pred_congestion = self.congestion_head(feat)
        
        return pred_flow, pred_congestion, attn_weights


if __name__ == "__main__":
    model = TrafficLSTM(input_dim=7, hidden_dim=64, forecast_horizon=3)
    dummy_x = torch.randn(8, 12, 7)  # 8 samples, 12 past steps (1 hour), 7 features
    flow_pred, cong_pred, weights = model(dummy_x)
    print("LSTM Architecture successfully verified:")
    print("  Input Shape:       ", dummy_x.shape)
    print("  Pred Flow Shape:   ", flow_pred.shape)
    print("  Pred Congestion:   ", cong_pred.shape)
    print("  Attn Weights Shape:", weights.shape)
