"""Index MIDI files: scan directory, validate, create manifest.

Creates a CSV manifest with metadata for each valid MIDI file:
- File path
- Duration
- Number of notes
- Estimated tempo
- Estimated key
- Train/val/test split assignment
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pretty_midi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_midi(midi_path: Path) -> Dict:
    """Validate and extract metadata from MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        Metadata dictionary or None if invalid
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        logger.debug(f"Failed to load {midi_path}: {e}")
        return None
    
    # Count notes
    num_notes = 0
    for instrument in midi.instruments:
        if not instrument.is_drum:
            num_notes += len(instrument.notes)
    
    # Skip if no notes
    if num_notes == 0:
        return None
    
    # Get duration
    duration = midi.get_end_time()
    
    if duration == 0:
        return None
    
    # Estimate tempo
    try:
        tempo = midi.estimate_tempo()
    except:
        tempo = 120.0
    
    # Simple key estimation (just most common pitch class)
    pitch_counts = [0] * 12
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch_counts[note.pitch % 12] += 1
    
    key_idx = np.argmax(pitch_counts) if any(pitch_counts) else 0
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = key_names[key_idx]
    
    return {
        'path': str(midi_path),
        'duration': duration,
        'num_notes': num_notes,
        'tempo': tempo,
        'key': key,
    }


def assign_splits(
    metadata_list: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """Assign train/val/test splits.
    
    Args:
        metadata_list: List of metadata dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Metadata list with 'split' field added
    """
    np.random.seed(seed)
    
    # Stratify by tempo buckets
    tempo_bins = [60, 90, 120, 150, 180]
    
    def get_tempo_bucket(tempo):
        for i, threshold in enumerate(tempo_bins):
            if tempo < threshold:
                return i
        return len(tempo_bins)
    
    # Group by tempo bucket
    buckets = {}
    for item in metadata_list:
        bucket = get_tempo_bucket(item['tempo'])
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(item)
    
    # Assign splits within each bucket
    for bucket_items in buckets.values():
        n = len(bucket_items)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for i, idx in enumerate(indices):
            if i < train_end:
                bucket_items[idx]['split'] = 'train'
            elif i < val_end:
                bucket_items[idx]['split'] = 'val'
            else:
                bucket_items[idx]['split'] = 'test'
    
    return metadata_list


def index_midi_directory(
    midi_dir: Path,
    output_csv: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Index all MIDI files in directory.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_csv: Path to output CSV manifest
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed for split
    """
    # Find all MIDI files
    midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    if not midi_files:
        logger.warning(f"No MIDI files found in {midi_dir}")
        return
    
    # Validate and extract metadata
    metadata_list = []
    
    for midi_path in tqdm(midi_files, desc="Validating MIDI files"):
        metadata = validate_midi(midi_path)
        if metadata:
            metadata_list.append(metadata)
    
    logger.info(f"Valid MIDI files: {len(metadata_list)}/{len(midi_files)}")
    
    if not metadata_list:
        logger.error("No valid MIDI files found")
        return
    
    # Assign splits
    metadata_list = assign_splits(metadata_list, train_ratio, val_ratio, seed)
    
    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['path', 'duration', 'num_notes', 'tempo', 'key', 'split']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    logger.info(f"Manifest saved to {output_csv}")
    
    # Print summary
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for item in metadata_list:
        split_counts[item['split']] += 1
    
    logger.info(f"Split summary: {split_counts}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Index MIDI files")
    parser.add_argument("--midi_dir", type=Path, required=True, help="Directory with MIDI files")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV manifest")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    index_midi_directory(
        args.midi_dir,
        args.out,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
