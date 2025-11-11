"""Neural network models for music generation.

Implements:
- TinyGRU: Small GRU-based model for CPU inference
- MiniTransformer: Lightweight transformer model
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class TinyGRU(nn.Module):
    """Small GRU model for next-token prediction."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialize TinyGRU.
        
        Args:
            vocab_size: Size of token vocabulary
            hidden_size: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch, seq_len]
            hidden: Hidden state [num_layers, batch, hidden_size]
            
        Returns:
            (logits, hidden) where logits is [batch, seq_len, vocab_size]
        """
        # Embedding
        emb = self.embedding(x)  # [batch, seq_len, hidden_size]
        emb = self.dropout(emb)
        
        # GRU
        out, hidden = self.gru(emb, hidden)  # out: [batch, seq_len, hidden_size]
        out = self.dropout(out)
        
        # Output projection
        logits = self.fc(out)  # [batch, seq_len, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1)]


class MiniTransformer(nn.Module):
    """Lightweight transformer for music generation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        """Initialize MiniTransformer.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch, seq_len]
            
        Returns:
            logits [batch, seq_len, vocab_size]
        """
        # Embedding
        emb = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        emb = self.pos_encoder(emb)
        emb = self.dropout(emb)
        
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transformer
        out = self.transformer(emb, mask=mask, is_causal=True)  # [batch, seq_len, d_model]
        out = self.dropout(out)
        
        # Output projection
        logits = self.fc(out)  # [batch, seq_len, vocab_size]
        
        return logits


def create_model(model_type: str, vocab_size: int, config: dict) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: "gru" or "transformer"
        vocab_size: Size of vocabulary
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    models_config = config.get('models', {})
    
    if model_type == 'gru':
        gru_config = models_config.get('gru', {})
        return TinyGRU(
            vocab_size=vocab_size,
            hidden_size=gru_config.get('hidden_size', 128),
            num_layers=gru_config.get('num_layers', 2),
            dropout=gru_config.get('dropout', 0.2),
        )
    elif model_type == 'transformer':
        tx_config = models_config.get('transformer', {})
        return MiniTransformer(
            vocab_size=vocab_size,
            d_model=tx_config.get('d_model', 256),
            nhead=tx_config.get('nhead', 2),
            num_layers=tx_config.get('num_layers', 2),
            dim_feedforward=tx_config.get('dim_feedforward', 512),
            dropout=tx_config.get('dropout', 0.2),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
