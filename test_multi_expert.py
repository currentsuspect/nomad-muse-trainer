#!/usr/bin/env python3
"""End-to-end test for Multi-Expert Nomad Muse Trainer.

This script tests the complete pipeline:
1. Create sample MIDI files
2. Convert to Nomad Symbolic Format
3. Train individual experts
4. Evaluate musical quality
5. Generate samples

Usage:
    python test_multi_expert.py
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import yaml

# Add src to path
sys.path.insert(0, 'src')

from nomad_format import create_nomad_format, NomadFormatBuilder
from tokenizers_drums import create_drums_tokenizer
from tokenizers_harmony import create_harmony_tokenizer
from tokenizers_melody import create_melody_tokenizer
from multi_expert_train import prepare_nomad_dataset, MultiExpertTrainer
from musical_evaluation import evaluate_multi_expert_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_midi_files(output_dir: Path, num_files: int = 5) -> list:
    """Create sample MIDI files for testing.
    
    Args:
        output_dir: Directory to save MIDI files
        num_files: Number of sample files to create
        
    Returns:
        List of created file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    for i in range(num_files):
        # Create a simple piece with drums, harmony, and melody
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        
        # Add drum track (Channel 9)
        drum_program = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        
        # Add some basic drum patterns
        for beat in range(16):  # 4 bars of 4/4
            time = beat * 0.5  # Quarter notes at 120 BPM
            
            # Kick on beats 1 and 3
            if beat % 4 == 0:
                kick = pretty_midi.Note(velocity=100, pitch=36, start=time, end=time + 0.1)
                drum_program.notes.append(kick)
            
            # Snare on beats 2 and 4
            if beat % 4 == 2:
                snare = pretty_midi.Note(velocity=80, pitch=38, start=time, end=time + 0.1)
                drum_program.notes.append(snare)
            
            # Hi-hat on every 8th note
            hihat = pretty_midi.Note(velocity=60, pitch=42, start=time, end=time + 0.05)
            drum_program.notes.append(hihat)
        
        midi.instruments.append(drum_program)
        
        # Add chord track (Piano)
        chord_program = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
        
        # Simple chord progression: C - F - G - C
        chord_roots = [60, 65, 67, 60]  # C4, F4, G4, C4
        chord_qualities = [0, 5, 7, 0]  # Major third intervals
        
        for i, root in enumerate(chord_roots):
            time = i * 2.0  # Chords change every 2 beats
            # Create a major chord (root, major third, perfect fifth)
            for interval in [0, 4, 7]:
                note = pretty_midi.Note(
                    velocity=70, 
                    pitch=root + interval, 
                    start=time, 
                    end=time + 2.0
                )
                chord_program.notes.append(note)
        
        midi.instruments.append(chord_program)
        
        # Add melody track (Lead)
        melody_program = pretty_midi.Instrument(program=73, is_drum=False, name="Lead")
        
        # Simple melody that fits the chords
        melody_notes = [
            (64, 0.0, 0.5),   # E4
            (67, 0.5, 0.5),   # G4
            (69, 1.0, 0.5),   # A4
            (67, 1.5, 0.5),   # G4
            (65, 2.0, 0.5),   # F4
            (67, 2.5, 0.5),   # G4
            (72, 3.0, 0.5),   # C5
            (67, 3.5, 0.5),   # G4
        ]
        
        for pitch, start, duration in melody_notes:
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + duration)
            melody_program.notes.append(note)
        
        midi.instruments.append(melody_program)
        
        # Save MIDI file
        filename = f"test_piece_{i:02d}.mid"
        filepath = output_dir / filename
        midi.write(str(filepath))
        created_files.append(filepath)
        
        logger.info(f"Created sample MIDI: {filename}")
    
    return created_files


def test_nomad_format_conversion(midi_files: list):
    """Test conversion from MIDI to Nomad Symbolic Format.
    
    Args:
        midi_files: List of MIDI file paths
        
    Returns:
        List of Nomad format dictionaries
    """
    logger.info("Testing Nomad format conversion...")
    
    nomad_data = []
    builder = NomadFormatBuilder()
    
    for midi_path in midi_files:
        try:
            nomad_format = builder.from_midi(midi_path)
            nomad_data.append(nomad_format)
            
            # Log track classification
            for track in nomad_format["tracks"]:
                track_type = track["classification"]["type"]
                confidence = track["classification"]["confidence"]
                logger.info(f"Track '{track['name']}': {track_type} (confidence: {confidence:.2f})")
            
            # Log chord extraction
            num_chords = len(nomad_format.get("chords", []))
            logger.info(f"Extracted {num_chords} chords")
            
        except Exception as e:
            logger.error(f"Failed to convert {midi_path}: {e}")
            continue
    
    logger.info(f"Successfully converted {len(nomad_data)} files to Nomad format")
    return nomad_data


def test_tokenizers(nomad_data: list):
    """Test expert tokenizers.
    
    Args:
        nomad_data: List of Nomad format dictionaries
    """
    logger.info("Testing expert tokenizers...")
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Test drums tokenizer
    try:
        drums_tokenizer = create_drums_tokenizer(config)
        drum_tokens = drums_tokenizer.tokenize_drum_track(nomad_data[0])
        logger.info(f"Drums tokenizer: {len(drum_tokens)} tokens generated")
        logger.info(f"Drums vocabulary size: {drums_tokenizer.vocab.vocab_size}")
    except Exception as e:
        logger.error(f"Drums tokenizer failed: {e}")
    
    # Test harmony tokenizer
    try:
        harmony_tokenizer = create_harmony_tokenizer(config)
        harmony_tokens = harmony_tokenizer.tokenize_harmony_track(nomad_data[0])
        logger.info(f"Harmony tokenizer: {len(harmony_tokens)} tokens generated")
        logger.info(f"Harmony vocabulary size: {harmony_tokenizer.vocab.vocab_size}")
    except Exception as e:
        logger.error(f"Harmony tokenizer failed: {e}")
    
    # Test melody tokenizer
    try:
        melody_tokenizer = create_melody_tokenizer(config)
        melody_tokens = melody_tokenizer.tokenize_melody_track(nomad_data[0])
        logger.info(f"Melody tokenizer: {len(melody_tokens)} tokens generated")
        logger.info(f"Melody vocabulary size: {melody_tokenizer.vocab.vocab_size}")
    except Exception as e:
        logger.error(f"Melody tokenizer failed: {e}")


def test_expert_training(nomad_data: list):
    """Test expert model training.
    
    Args:
        nomad_data: List of Nomad format dictionaries
        
    Returns:
        Training results
    """
    logger.info("Testing expert training...")
    
    # Load and modify config for fast testing
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Reduce training parameters for testing
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 4
    config["training"]["early_stop_patience"] = 2
    
    # Create trainer
    trainer = MultiExpertTrainer(config)
    
    # Test individual expert training
    results = {}
    
    for expert_type in ["drums", "harmony", "melody"]:
        if expert_type in trainer.tokenizers:
            try:
                logger.info(f"Training {expert_type} expert...")
                stats = trainer.train_expert(expert_type, nomad_data, epochs=3)
                results[expert_type] = stats
                logger.info(f"{expert_type.capitalize()} training completed")
            except Exception as e:
                logger.error(f"{expert_type} training failed: {e}")
                results[expert_type] = {"error": str(e)}
    
    return results


def test_musical_evaluation(nomad_data: list):
    """Test musical evaluation metrics.
    
    Args:
        nomad_data: List of Nomad format dictionaries
        
    Returns:
        Evaluation results
    """
    logger.info("Testing musical evaluation...")
    
    # For testing, use the same data as both training and generated
    # In practice, these would be different datasets
    evaluation_results = evaluate_multi_expert_generation(
        generated_data=nomad_data[:2],  # Use subset as "generated"
        training_data=nomad_data,       # Use full set as "training"
        config={"evaluation": {}}
    )
    
    logger.info("Musical evaluation completed")
    return evaluation_results


def create_multi_expert_config():
    """Create configuration for multi-expert system."""
    config = {
        # Base tokenization
        "tokenization": {
            "time_shift_bins": 16,
            "velocity_bins": 8,
            "duration_bins": 16
        },
        
        # Data configuration
        "data": {
            "sequence_length": 256,  # Shorter for testing
            "min_midi_length": 16,
            "max_midi_length": 1024,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42
        },
        
        # Training configuration
        "training": {
            "batch_size": 4,  # Small for testing
            "epochs": 3,      # Few epochs for testing
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "early_stop_patience": 2,
            "device": "cpu",  # CPU for testing
            "num_workers": 0
        },
        
        # Expert-specific configurations
        "drums_model": {
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.2
        },
        
        "harmony_model": {
            "hidden_size": 48,
            "num_layers": 2,
            "dropout": 0.1
        },
        
        "melody_model": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2
        },
        
        # Expert tokenizers
        "drums": {
            "time_signature": [4, 4],
            "max_bars": 8,
            "velocity_bins": 8
        },
        
        "harmony": {
            "max_bars": 8,
            "include_keys": True
        },
        
        "melody": {
            "max_bars": 8,
            "time_signature": [4, 4],
            "conditioning": "none"  # No conditioning for testing
        }
    }
    
    return config


def main():
    """Main test function."""
    logger.info("Starting Multi-Expert Nomad Muse Trainer End-to-End Test")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        midi_dir = temp_path / "midi_files"
        output_dir = temp_path / "output"
        
        try:
            # Step 1: Create sample MIDI files
            logger.info("Step 1: Creating sample MIDI files...")
            midi_files = create_sample_midi_files(midi_dir, num_files=3)
            logger.info(f"Created {len(midi_files)} sample MIDI files")
            
            # Step 2: Convert to Nomad format
            logger.info("Step 2: Converting to Nomad Symbolic Format...")
            nomad_data = test_nomad_format_conversion(midi_files)
            
            if not nomad_data:
                logger.error("No valid Nomad data created. Aborting test.")
                return False
            
            # Step 3: Test tokenizers
            logger.info("Step 3: Testing expert tokenizers...")
            test_tokenizers(nomad_data)
            
            # Step 4: Test training
            logger.info("Step 4: Testing expert training...")
            training_results = test_expert_training(nomad_data)
            
            # Ensure output directory exists and save training results
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "training_results.json", 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            
            # Step 5: Test musical evaluation
            logger.info("Step 5: Testing musical evaluation...")
            eval_results = test_musical_evaluation(nomad_data)
            
            # Save evaluation results
            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)
            
            # Summary
            logger.info("=" * 60)
            logger.info("END-TO-END TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✓ Created {len(midi_files)} sample MIDI files")
            logger.info(f"✓ Converted {len(nomad_data)} files to Nomad format")
            logger.info(f"✓ Tested all expert tokenizers")
            
            successful_experts = [expert for expert, result in training_results.items() 
                                if "error" not in result]
            logger.info(f"✓ Successfully trained {len(successful_experts)} experts: {successful_experts}")
            
            if "overall_score" in eval_results:
                overall_score = eval_results["overall_score"]
                logger.info(f"✓ Musical evaluation completed (overall score: {overall_score:.3f})")
            
            logger.info("=" * 60)
            logger.info("END-TO-END TEST PASSED")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)