"""
This file defines a Transformer-based model for processing sequential data.

Model Architecture: Feature Extraction -> Transformer Enhancement -> Aggregation -> Prediction

Feature: This is the "compatibility mode" version. By preserving the original
         ModuleList structure, it can load weight files (.pth) that were trained
         using the older, custom Transformer layer implementation.
"""

import torch
import torch.nn as nn
import math

# ==================== 1. Configuration Parameters ====================
class Config:
    batch_size = 512
    input_dim = 2
    output_size = 100
    feature_dim = 128
    hidden_size = 256
    d_model = 128
    nhead = 8
    num_transformer_layers = 6
    dim_feedforward = 1024
    dropout = 0
    feature_dropout = 0
    predictor_dropout = 0

# ==================== 2. Model Module Definitions ====================

class FeatureExtractor(nn.Module):
    """
    Extracts high-dimensional features for each point in the sequence independently.
    """
    def __init__(self, input_dim=2, feature_dim=128, dropout=0.1):
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
    def forward(self, x):
        return self.layers(x)

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input features.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEnhancer(nn.Module):
    """
    Enhances features using a multi-layer Transformer Encoder to capture sequential dependencies.
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEnhancer, self).__init__()
        if num_layers > 0:
            self.pos_encoding = PositionalEncoding(d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                ) for _ in range(num_layers)
            ])
        else:
            self.layers = None

    def forward(self, x):
        if self.layers is not None:
            x = self.pos_encoding(x)
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            return x

class Predictor(nn.Module):
    """
    Generates the final prediction based on the aggregated feature vector.
    """
    def __init__(self, feature_dim=128, hidden_size=256, output_size=100, dropout=0.1):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = self.activation(x)
        return x * 1000
# ==================== 3. Main Model Wrapper ====================

class LayerNormSumAggregationModel(nn.Module):
    """
    The final model that combines all the preceding modules.
    """
    def __init__(self, config, use_transformer=True):
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
                dropout=config.dropout
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        point_features = self.feature_extractor(x)
        
        if self.transformer_enhancer is not None:
            enhanced_features = self.transformer_enhancer(point_features)
        else:
            enhanced_features = point_features
            
        aggregated_features = torch.sum(enhanced_features, dim=1)
        predictions = self.predictor(aggregated_features)
        return predictions