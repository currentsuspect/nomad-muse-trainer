"""Vocabulary builder for music tokenization.

Defines a discrete vocabulary for symbolic music events:
- NOTE_ON_<pitch>: Note starts (0-127)
- NOTE_OFF_<pitch>: Note ends (0-127)
- TIME_SHIFT_<bin>: Time advancement
- VELOCITY_<bin>: Velocity quantization
- DURATION_<bin>: Duration quantization
- Special tokens: PAD, BOS, EOS, UNK
"""

import json
from typing import Dict, List, Tuple
from pathlib import Path


class MusicVocabulary:
    """Manages discrete token vocabulary for symbolic music."""
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(
        self,
        time_shift_bins: int = 32,
        velocity_bins: int = 8,
        duration_bins: int = 16,
    ):
        """Initialize vocabulary.
        
        Args:
            time_shift_bins: Number of discrete time shift tokens
            velocity_bins: Number of velocity quantization levels
            duration_bins: Number of duration quantization levels
        """
        self.time_shift_bins = time_shift_bins
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        self._build_vocab()
    
    def _build_vocab(self):
        """Build the complete vocabulary."""
        tokens = []
        
        # Special tokens
        tokens.extend([self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])
        
        # Note on/off for each MIDI pitch (0-127)
        for pitch in range(128):
            tokens.append(f"NOTE_ON_{pitch}")
        for pitch in range(128):
            tokens.append(f"NOTE_OFF_{pitch}")
        
        # Time shifts
        for i in range(1, self.time_shift_bins + 1):
            tokens.append(f"TIME_SHIFT_{i}")
        
        # Velocity bins
        for i in range(1, self.velocity_bins + 1):
            tokens.append(f"VELOCITY_{i}")
        
        # Duration bins
        for i in range(1, self.duration_bins + 1):
            tokens.append(f"DURATION_{i}")
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def encode(self, token: str) -> int:
        """Convert token string to ID."""
        return self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
    
    def decode(self, token_id: int) -> str:
        """Convert token ID to string."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)
    
    def encode_sequence(self, tokens: List[str]) -> List[int]:
        """Encode a sequence of tokens."""
        return [self.encode(token) for token in tokens]
    
    def decode_sequence(self, token_ids: List[int]) -> List[str]:
        """Decode a sequence of token IDs."""
        return [self.decode(tid) for tid in token_ids]
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self.token_to_id)
    
    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def bos_id(self) -> int:
        """Begin-of-sequence token ID."""
        return self.token_to_id[self.BOS_TOKEN]
    
    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.token_to_id[self.EOS_TOKEN]
    
    def quantize_velocity(self, velocity: int) -> str:
        """Quantize MIDI velocity (0-127) into bins.
        
        Args:
            velocity: MIDI velocity value
            
        Returns:
            Velocity token (e.g., "VELOCITY_4")
        """
        if velocity <= 0:
            bin_idx = 1
        else:
            bin_idx = min(int((velocity / 127.0) * self.velocity_bins) + 1, self.velocity_bins)
        return f"VELOCITY_{bin_idx}"
    
    def quantize_duration(self, duration_ms: float, max_duration_ms: float = 2000) -> str:
        """Quantize duration in milliseconds into bins.
        
        Args:
            duration_ms: Duration in milliseconds
            max_duration_ms: Maximum duration to consider
            
        Returns:
            Duration token (e.g., "DURATION_8")
        """
        duration_ms = min(duration_ms, max_duration_ms)
        bin_idx = max(1, min(int((duration_ms / max_duration_ms) * self.duration_bins) + 1, self.duration_bins))
        return f"DURATION_{bin_idx}"
    
    def quantize_time_shift(self, time_ms: float, time_shift_ms: float = 20) -> str:
        """Quantize time shift into discrete bins.
        
        Args:
            time_ms: Time in milliseconds
            time_shift_ms: Milliseconds per bin
            
        Returns:
            Time shift token (e.g., "TIME_SHIFT_5")
        """
        bins = max(1, min(int(time_ms / time_shift_ms), self.time_shift_bins))
        return f"TIME_SHIFT_{bins}"
    
    def save(self, path: Path):
        """Save vocabulary to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "config": {
                "time_shift_bins": self.time_shift_bins,
                "velocity_bins": self.velocity_bins,
                "duration_bins": self.duration_bins,
            }
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'MusicVocabulary':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        config = vocab_data["config"]
        vocab = cls(
            time_shift_bins=config["time_shift_bins"],
            velocity_bins=config["velocity_bins"],
            duration_bins=config["duration_bins"],
        )
        
        return vocab


def create_vocab_from_config(config: dict) -> MusicVocabulary:
    """Create vocabulary from configuration dictionary.
    
    Args:
        config: Configuration dict with tokenization parameters
        
    Returns:
        MusicVocabulary instance
    """
    tok_config = config.get("tokenization", {})
    return MusicVocabulary(
        time_shift_bins=tok_config.get("time_shift_bins", 32),
        velocity_bins=tok_config.get("velocity_bins", 8),
        duration_bins=tok_config.get("duration_bins", 16),
    )
