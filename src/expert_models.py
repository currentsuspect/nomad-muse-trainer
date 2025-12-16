"""Multi-Expert Models for Nomad Muse Trainer.

Implements specialized models for different musical domains:
- MuseDrums: Percussion generation
- MuseHarmony: Chord progression generation
- MuseMelody: Melodic generation with harmony conditioning

Each expert follows the "v1: GRU fast start" approach but with domain-specific
architectures optimized for their musical role.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from model import TinyGRU, MiniTransformer


class ExpertEmbedding(nn.Module):
    """Shared embedding layer for expert models."""
    
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        return self.dropout(emb)


class MuseDrumsModel(nn.Module):
    """Expert model for drum pattern generation.
    
    Optimized for:
    - Fast inference (GRU-based)
    - Groove consistency
    - Pattern variation
    - Fill generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,  # Smaller for drums
        num_layers: int = 2,
        dropout: float = 0.2,
        groove_memory: int = 16  # Bars of groove memory
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.groove_memory = groove_memory
        
        # Embedding
        self.embedding = ExpertEmbedding(vocab_size, hidden_size, dropout)
        
        # Main GRU for sequence modeling
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Groove consistency head
        self.groove_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Variation head for pattern diversity
        self.variation_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        groove_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch, seq_len]
            hidden: Hidden state [num_layers, batch, hidden_size]
            groove_context: Groove memory context [batch, groove_memory, hidden_size]
            
        Returns:
            (logits, hidden) where logits is [batch, seq_len, vocab_size]
        """
        # Embedding
        emb = self.embedding(x)  # [batch, seq_len, hidden_size]
        
        # Add groove context if provided
        if groove_context is not None:
            # Average groove context and add to embedding
            groove_avg = torch.mean(groove_context, dim=1, keepdim=True)  # [batch, 1, hidden_size]
            emb = emb + groove_avg
        
        # GRU
        out, hidden = self.gru(emb, hidden)  # [batch, seq_len, hidden_size]
        out = self.dropout(out)
        
        # Multi-head output for groove consistency and variation
        groove_features = torch.tanh(self.groove_head(out))
        variation_features = torch.tanh(self.variation_head(out))
        
        # Combine features
        combined_features = torch.cat([groove_features, variation_features], dim=-1)
        
        # Project to vocabulary
        logits = self.output_proj(combined_features)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    @property
    def num_layers(self) -> int:
        return self.gru.num_layers


class MuseHarmonyModel(nn.Module):
    """Expert model for chord progression generation.
    
    Optimized for:
    - Stable progressions
    - Key awareness
    - Harmonic rhythm
    - Voice leading
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 96,  # Medium size for harmony
        num_layers: int = 3,
        dropout: float = 0.1,
        num_keys: int = 24,  # 12 major + 12 minor
        chord_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_keys = num_keys
        self.chord_embedding_dim = chord_embedding_dim
        
        # Token embedding
        self.token_embedding = ExpertEmbedding(vocab_size, hidden_size, dropout)
        
        # Key embedding for harmonic context
        self.key_embedding = nn.Embedding(num_keys, chord_embedding_dim)
        
        # Chord quality embedding
        self.chord_embedding = nn.Embedding(8, chord_embedding_dim)  # maj, min, dom7, min7, maj7, dim, sus2, sus4
        
        # Main GRU
        self.gru = nn.GRU(
            hidden_size + chord_embedding_dim * 2,  # token + key + chord embeddings
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Harmonic stability head
        self.stability_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Voice leading head
        self.voice_leading_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        key_context: Optional[torch.Tensor] = None,
        chord_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch, seq_len]
            hidden: Hidden state [num_layers, batch, hidden_size]
            key_context: Key context [batch, seq_len, chord_embedding_dim]
            chord_context: Chord context [batch, seq_len, chord_embedding_dim]
            
        Returns:
            (logits, hidden)
        """
        # Token embedding
        token_emb = self.token_embedding(x)  # [batch, seq_len, hidden_size]
        
        # Combine embeddings
        if key_context is not None and chord_context is not None:
            combined_emb = torch.cat([token_emb, key_context, chord_context], dim=-1)
        elif key_context is not None:
            combined_emb = torch.cat([token_emb, key_context], dim=-1)
        elif chord_context is not None:
            combined_emb = torch.cat([token_emb, chord_context], dim=-1)
        else:
            combined_emb = token_emb
        
        # GRU
        out, hidden = self.gru(combined_emb, hidden)
        out = self.dropout(out)
        
        # Multi-head output
        stability_features = torch.tanh(self.stability_head(out))
        voice_leading_features = torch.tanh(self.voice_leading_head(out))
        
        # Combine features
        combined_features = torch.cat([stability_features, voice_leading_features], dim=-1)
        
        # Project to vocabulary
        logits = self.output_proj(combined_features)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    @property
    def num_layers(self) -> int:
        return self.gru.num_layers


class MuseMelodyModel(nn.Module):
    """Expert model for melodic generation with harmony conditioning.
    
    Optimized for:
    - Coherent phrases
    - Chord-tone relationships
    - Melodic contour
    - Harmony conditioning
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,  # Larger for melody complexity
        num_layers: int = 3,
        dropout: float = 0.2,
        conditioning_dim: int = 64,
        pitch_range: int = 128
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.conditioning_dim = conditioning_dim
        self.pitch_range = pitch_range
        
        # Token embedding
        self.token_embedding = ExpertEmbedding(vocab_size, hidden_size, dropout)
        
        # Pitch embedding for note events
        self.pitch_embedding = nn.Embedding(pitch_range, hidden_size // 4)
        
        # Conditioning embeddings
        self.chord_conditioning = nn.Linear(12 * 8, conditioning_dim)  # chord root + quality
        self.key_conditioning = nn.Linear(12 * 2, conditioning_dim)    # key root + mode
        
        # Main GRU
        self.gru = nn.GRU(
            hidden_size + conditioning_dim + hidden_size // 4,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Phrase coherence head
        self.phrase_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Chord-tone relationship head
        self.chord_tone_head = nn.Linear(hidden_size, hidden_size // 2)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        chord_condition: Optional[torch.Tensor] = None,
        key_condition: Optional[torch.Tensor] = None,
        pitch_condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input token IDs [batch, seq_len]
            hidden: Hidden state [num_layers, batch, hidden_size]
            chord_condition: Chord conditioning [batch, seq_len, conditioning_dim]
            key_condition: Key conditioning [batch, seq_len, conditioning_dim]
            pitch_condition: Pitch conditioning [batch, seq_len, hidden_size//4]
            
        Returns:
            (logits, hidden)
        """
        # Token embedding
        token_emb = self.token_embedding(x)  # [batch, seq_len, hidden_size]
        
        # Combine conditioning
        conditioning_features = []
        if chord_condition is not None:
            conditioning_features.append(chord_condition)
        if key_condition is not None:
            conditioning_features.append(key_condition)
        
        if conditioning_features:
            combined_conditioning = torch.cat(conditioning_features, dim=-1)
        else:
            combined_conditioning = torch.zeros_like(token_emb)
        
        # Add pitch conditioning if available
        if pitch_condition is not None:
            combined_input = torch.cat([token_emb, combined_conditioning, pitch_condition], dim=-1)
        else:
            combined_input = torch.cat([token_emb, combined_conditioning], dim=-1)
        
        # GRU
        out, hidden = self.gru(combined_input, hidden)
        out = self.dropout(out)
        
        # Multi-head output
        phrase_features = torch.tanh(self.phrase_head(out))
        chord_tone_features = torch.tanh(self.chord_tone_head(out))
        
        # Combine features
        combined_features = torch.cat([phrase_features, chord_tone_features], dim=-1)
        
        # Project to vocabulary
        logits = self.output_proj(combined_features)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    @property
    def num_layers(self) -> int:
        return self.gru.num_layers


class MultiExpertModel:
    """Container for multiple expert models with coordinated inference."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-expert system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize expert models
        self.drums_model = None
        self.harmony_model = None
        self.melody_model = None
        
        # Model states
        self.training = True
    
    def build_experts(self, vocabularies: Dict[str, int]):
        """Build all expert models.
        
        Args:
            vocabularies: Dict of expert_name -> vocab_size
        """
        drums_config = self.config.get("drums_model", {})
        harmony_config = self.config.get("harmony_model", {})
        melody_config = self.config.get("melody_model", {})
        
        # Build drums model
        if "drums" in vocabularies:
            self.drums_model = MuseDrumsModel(
                vocab_size=vocabularies["drums"],
                hidden_size=drums_config.get("hidden_size", 64),
                num_layers=drums_config.get("num_layers", 2),
                dropout=drums_config.get("dropout", 0.2)
            )
        
        # Build harmony model
        if "harmony" in vocabularies:
            self.harmony_model = MuseHarmonyModel(
                vocab_size=vocabularies["harmony"],
                hidden_size=harmony_config.get("hidden_size", 96),
                num_layers=harmony_config.get("num_layers", 3),
                dropout=harmony_config.get("dropout", 0.1)
            )
        
        # Build melody model
        if "melody" in vocabularies:
            self.melody_model = MuseMelodyModel(
                vocab_size=vocabularies["melody"],
                hidden_size=melody_config.get("hidden_size", 128),
                num_layers=melody_config.get("num_layers", 3),
                dropout=melody_config.get("dropout", 0.2)
            )
    
    def train(self):
        """Set all models to training mode."""
        self.training = True
        if self.drums_model:
            self.drums_model.train()
        if self.harmony_model:
            self.harmony_model.train()
        if self.melody_model:
            self.melody_model.train()
    
    def eval(self):
        """Set all models to evaluation mode."""
        self.training = False
        if self.drums_model:
            self.drums_model.eval()
        if self.harmony_model:
            self.harmony_model.eval()
        if self.melody_model:
            self.melody_model.eval()
    
    def parameters(self):
        """Get all parameters from expert models."""
        params = []
        if self.drums_model:
            params.extend(list(self.drums_model.parameters()))
        if self.harmony_model:
            params.extend(list(self.harmony_model.parameters()))
        if self.melody_model:
            params.extend(list(self.melody_model.parameters()))
        return params
    
    def named_parameters(self):
        """Get named parameters from expert models."""
        named_params = []
        if self.drums_model:
            named_params.extend(self.drums_model.named_parameters())
        if self.harmony_model:
            named_params.extend(self.harmony_model.named_parameters())
        if self.melody_model:
            named_params.extend(self.melody_model.named_parameters())
        return named_params
    
    def save_state_dict(self, path: str):
        """Save expert model state dictionaries."""
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.drums_model:
            torch.save(self.drums_model.state_dict(), f"{path}/drums_model.pt")
        if self.harmony_model:
            torch.save(self.harmony_model.state_dict(), f"{path}/harmony_model.pt")
        if self.melody_model:
            torch.save(self.melody_model.state_dict(), f"{path}/melody_model.pt")
    
    def load_state_dict(self, path: str):
        """Load expert model state dictionaries."""
        import os
        
        if self.drums_model and os.path.exists(f"{path}/drums_model.pt"):
            self.drums_model.load_state_dict(torch.load(f"{path}/drums_model.pt"))
        if self.harmony_model and os.path.exists(f"{path}/harmony_model.pt"):
            self.harmony_model.load_state_dict(torch.load(f"{path}/harmony_model.pt"))
        if self.melody_model and os.path.exists(f"{path}/melody_model.pt"):
            self.melody_model.load_state_dict(torch.load(f"{path}/melody_model.pt"))


def create_expert_model(expert_type: str, vocab_size: int, config: Dict[str, Any]) -> nn.Module:
    """Factory function to create expert models.
    
    Args:
        expert_type: "drums", "harmony", or "melody"
        vocab_size: Size of vocabulary
        config: Configuration dictionary
        
    Returns:
        Expert model instance
    """
    expert_config = config.get(f"{expert_type}_model", {})
    
    if expert_type == "drums":
        return MuseDrumsModel(
            vocab_size=vocab_size,
            hidden_size=expert_config.get("hidden_size", 64),
            num_layers=expert_config.get("num_layers", 2),
            dropout=expert_config.get("dropout", 0.2)
        )
    elif expert_type == "harmony":
        return MuseHarmonyModel(
            vocab_size=vocab_size,
            hidden_size=expert_config.get("hidden_size", 96),
            num_layers=expert_config.get("num_layers", 3),
            dropout=expert_config.get("dropout", 0.1)
        )
    elif expert_type == "melody":
        return MuseMelodyModel(
            vocab_size=vocab_size,
            hidden_size=expert_config.get("hidden_size", 128),
            num_layers=expert_config.get("num_layers", 3),
            dropout=expert_config.get("dropout", 0.2)
        )
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")


def count_expert_parameters(model: nn.Module) -> int:
    """Count parameters in an expert model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)