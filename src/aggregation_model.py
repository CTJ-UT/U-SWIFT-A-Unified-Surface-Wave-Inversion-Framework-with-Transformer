"""
This file defines a Transformer-based model for processing sequential data.

Model Architecture: Feature Extraction -> Transformer Enhancement -> Aggregation -> Prediction

Uses PyTorch native nn.TransformerEncoder (norm_first=True, Pre-LN).
Predictor outputs sigmoid(x) without * 1000, so predictions are in normalized
space (vs / vs_hs). Use denormalize_dc to recover physical Vs values.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

# ==================== 1. Configuration Parameters ====================
class Config:
    batch_size: int = 512
    input_dim: int = 2
    output_size: int = 100
    feature_dim: int = 128
    hidden_size: int = 256
    d_model: int = 128
    nhead: int = 8
    num_transformer_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0
    feature_dropout: float = 0
    predictor_dropout: float = 0
    max_len: int = 500

# ==================== 2. Model Module Definitions ====================

class FeatureExtractor(nn.Module):
    """
    Extracts high-dimensional features for each point in the sequence independently.
    """
    def __init__(self, input_dim: int = 2, feature_dim: int = 128, dropout: float = 0.1) -> None:
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input features.
    """
    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f'Input curve length {x.size(1)} exceeds positional encoding max_len={self.pe.size(1)}. '
                f'Please increase Config.max_len.'
            )
        return x + self.pe[:, :x.size(1)]

class TransformerEnhancer(nn.Module):
    """
    Enhances features using a multi-layer Transformer Encoder to capture sequential dependencies.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super(TransformerEnhancer, self).__init__()
        self.use_layers = num_layers > 0
        if self.use_layers:
            self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layers:
            x = self.pos_encoding(x)
            return self.transformer_encoder(x)
        return x

class Predictor(nn.Module):
    """
    Generates the final prediction based on the aggregated feature vector.
    """
    def __init__(self, feature_dim: int = 128, hidden_size: int = 256, output_size: int = 100, dropout: float = 0.1) -> None:
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        return self.activation(x)
# ==================== 3. Main Model Wrapper ====================

class LayerNormSumAggregationModel(nn.Module):
    """
    The final model that combines all the preceding modules.
    """
    def __init__(self, config: Config, use_transformer: bool = True) -> None:
        super(LayerNormSumAggregationModel, self).__init__()
        self.use_transformer = use_transformer
        
        self.feature_extractor = FeatureExtractor(
            input_dim=config.input_dim,
            feature_dim=config.feature_dim,
            dropout=config.feature_dropout
        )
        
        if use_transformer:
            self.transformer_enhancer = TransformerEnhancer(
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_transformer_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                max_len=config.max_len
            )
        else:
            self.transformer_enhancer = None
            
        self.predictor = Predictor(
            feature_dim=config.feature_dim,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            dropout=config.predictor_dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_features = self.feature_extractor(x)
        
        if self.transformer_enhancer is not None:
            enhanced_features = self.transformer_enhancer(point_features)
        else:
            enhanced_features = point_features
            
        aggregated_features = torch.sum(enhanced_features, dim=1)
        predictions = self.predictor(aggregated_features)
        return predictions
