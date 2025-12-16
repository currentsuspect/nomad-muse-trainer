#!/usr/bin/env python3
"""Simplified end-to-end test for Multi-Expert Nomad Muse Trainer.

This test verifies the core functionality works correctly.
"""

import json
import logging
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi

# Add src to path
import sys
sys.path.insert(0, 'src')

from nomad_format import create_nomad_format, NomadFormatBuilder
from tokenizers_drums import create_drums_tokenizer, MuseDrumsTokenizer
from tokenizers_harmony import create_harmony_tokenizer, MuseHarmonyTokenizer  
from tokenizers_melody import create_melody_tokenizer, MuseMelodyTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_midi(output_path: Path) -> bool:
    """Create a simple test MIDI file."""
    try:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        
        # Add drum track
        drum_program = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        # Simple kick-snare pattern
        kick = pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1)
        snare = pretty_midi.Note(velocity=80, pitch=38, start=1.0, end=1.1)
        drum_program.notes.extend([kick, snare])
        midi.instruments.append(drum_program)
        
        # Add chord track
        chord_program = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
        # Simple C major chord
        for interval in [0, 4, 7]:  # C, E, G
            note = pretty_midi.Note(velocity=70, pitch=60+interval, start=0.0, end=2.0)
            chord_program.notes.append(note)
        midi.instruments.append(chord_program)
        
        # Add melody track
        melody_program = pretty_midi.Instrument(program=73, is_drum=False, name="Lead")
        # Simple melody
        for i, pitch in enumerate([64, 67, 69, 67]):  # E, G, A, G
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=i*0.5, end=(i+1)*0.5)
            melody_program.notes.append(note)
        midi.instruments.append(melody_program)
        
        midi.write(str(output_path))
        return True
        
    except Exception as e:
        logger.error(f"Failed to create MIDI: {e}")
        return False


def test_nomad_format():
    """Test conversion to Nomad Symbolic Format."""
    logger.info("Testing Nomad format conversion...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = Path(temp_dir) / "test.mid"
        
        # Create test MIDI
        if not create_simple_midi(midi_path):
            return False
        
        # Convert to Nomad format
        try:
            nomad_data = create_nomad_format(midi_path)
            
            # Verify structure
            assert "global" in nomad_data
            assert "tracks" in nomad_data
            assert "metadata" in nomad_data
            
            # Check track classification
            track_types = [track["classification"]["type"] for track in nomad_data["tracks"]]
            logger.info(f"Found track types: {track_types}")
            
            # Check chord extraction
            num_chords = len(nomad_data.get("chords", []))
            logger.info(f"Extracted {num_chords} chords")
            
            logger.info("✓ Nomad format conversion successful")
            return True
            
        except Exception as e:
            logger.error(f"Nomad format conversion failed: {e}")
            return False


def test_tokenizers():
    """Test expert tokenizers."""
    logger.info("Testing expert tokenizers...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = Path(temp_dir) / "test.mid"
        
        if not create_simple_midi(midi_path):
            return False
        
        # Convert to Nomad format
        nomad_data = create_nomad_format(midi_path)
        
        # Create simple config
        config = {
            "tokenization": {"velocity_bins": 8, "duration_bins": 16},
            "drums": {"time_signature": [4, 4], "max_bars": 8},
            "harmony": {"max_bars": 8, "include_keys": True},
            "melody": {"max_bars": 8, "time_signature": [4, 4], "conditioning": "none"}
        }
        
        # Test drums tokenizer
        try:
            drums_tokenizer = create_drums_tokenizer(config)
            drum_tokens = drums_tokenizer.tokenize_drum_track(nomad_data)
            logger.info(f"✓ Drums tokenizer: {len(drum_tokens)} tokens")
        except Exception as e:
            logger.error(f"Drums tokenizer failed: {e}")
            return False
        
        # Test harmony tokenizer
        try:
            harmony_tokenizer = create_harmony_tokenizer(config)
            harmony_tokens = harmony_tokenizer.tokenize_harmony_track(nomad_data)
            logger.info(f"✓ Harmony tokenizer: {len(harmony_tokens)} tokens")
        except Exception as e:
            logger.error(f"Harmony tokenizer failed: {e}")
            return False
        
        # Test melody tokenizer
        try:
            melody_tokenizer = create_melody_tokenizer(config)
            melody_tokens = melody_tokenizer.tokenize_melody_track(nomad_data)
            logger.info(f"✓ Melody tokenizer: {len(melody_tokens)} tokens")
        except Exception as e:
            logger.error(f"Melody tokenizer failed: {e}")
            return False
        
        logger.info("✓ All tokenizers working")
        return True


def test_round_trip():
    """Test tokenization round-trip."""
    logger.info("Testing tokenization round-trip...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        midi_path = Path(temp_dir) / "test.mid"
        
        if not create_simple_midi(midi_path):
            return False
        
        nomad_data = create_nomad_format(midi_path)
        
        config = {
            "tokenization": {"velocity_bins": 8, "duration_bins": 16},
            "drums": {"time_signature": [4, 4], "max_bars": 8},
        }
        
        try:
            drums_tokenizer = create_drums_tokenizer(config)
            
            # Tokenize
            tokens = drums_tokenizer.tokenize_drum_track(nomad_data)
            
            # Decode back
            events = drums_tokenizer.decode_to_midi_events(tokens)
            
            logger.info(f"✓ Round-trip: {len(tokens)} tokens → {len(events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Round-trip test failed: {e}")
            return False


def main():
    """Main test function."""
    logger.info("Starting Multi-Expert Nomad Muse Trainer Simplified Test")
    
    tests = [
        ("Nomad Format", test_nomad_format),
        ("Tokenizers", test_tokenizers),
        ("Round-trip", test_round_trip),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Multi-Expert system is working!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)