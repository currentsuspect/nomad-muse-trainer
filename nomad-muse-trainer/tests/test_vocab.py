"""Tests for vocabulary module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.vocab import MusicVocabulary


def test_vocab_creation():
    """Test vocabulary creation."""
    vocab = MusicVocabulary(
        time_shift_bins=32,
        velocity_bins=8,
        duration_bins=16,
    )
    
    # Check vocab size
    # 4 special + 128 NOTE_ON + 128 NOTE_OFF + 32 TIME_SHIFT + 8 VELOCITY + 16 DURATION
    expected_size = 4 + 128 + 128 + 32 + 8 + 16
    assert vocab.vocab_size == expected_size
    
    # Check special tokens
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2


def test_encode_decode():
    """Test encoding and decoding."""
    vocab = MusicVocabulary()
    
    # Test single token
    token = "NOTE_ON_60"
    token_id = vocab.encode(token)
    decoded = vocab.decode(token_id)
    assert decoded == token
    
    # Test sequence
    tokens = ["<BOS>", "NOTE_ON_60", "VELOCITY_4", "TIME_SHIFT_8", "<EOS>"]
    ids = vocab.encode_sequence(tokens)
    decoded_tokens = vocab.decode_sequence(ids)
    assert decoded_tokens == tokens


def test_quantization():
    """Test velocity, duration, and time quantization."""
    vocab = MusicVocabulary(velocity_bins=8, duration_bins=16)
    
    # Test velocity quantization
    vel_token = vocab.quantize_velocity(64)
    assert vel_token.startswith("VELOCITY_")
    
    # Test duration quantization
    dur_token = vocab.quantize_duration(500, max_duration_ms=2000)
    assert dur_token.startswith("DURATION_")
    
    # Test time shift quantization
    time_token = vocab.quantize_time_shift(40, time_shift_ms=20)
    assert time_token.startswith("TIME_SHIFT_")


def test_save_load():
    """Test saving and loading vocabulary."""
    vocab = MusicVocabulary(
        time_shift_bins=32,
        velocity_bins=8,
        duration_bins=16,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.json"
        
        # Save
        vocab.save(path)
        assert path.exists()
        
        # Load
        loaded_vocab = MusicVocabulary.load(path)
        
        # Verify
        assert loaded_vocab.vocab_size == vocab.vocab_size
        assert loaded_vocab.time_shift_bins == vocab.time_shift_bins
        assert loaded_vocab.velocity_bins == vocab.velocity_bins
        assert loaded_vocab.duration_bins == vocab.duration_bins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
