"""Data preparation: MIDI files to tokenized sequences.

Converts MIDI files to token sequences using pretty_midi, with:
- Tempo and key detection
- Multi-track support (optional melody-only mode)
- Train/val/test stratified splitting
- Deduplication and length filtering
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .vocab import MusicVocabulary, create_vocab_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIDITokenizer:
    """Convert MIDI files to token sequences."""
    
    def __init__(
        self,
        vocab: MusicVocabulary,
        time_shift_ms: float = 20,
        max_duration_ms: float = 2000,
        melody_only: bool = False,
    ):
        """Initialize tokenizer.
        
        Args:
            vocab: Music vocabulary
            time_shift_ms: Milliseconds per time shift unit
            max_duration_ms: Maximum duration to quantize
            melody_only: If True, only process first track
        """
        self.vocab = vocab
        self.time_shift_ms = time_shift_ms
        self.max_duration_ms = max_duration_ms
        self.melody_only = melody_only
    
    def tokenize_midi(self, midi_path):
        """
        Tokenize a MIDI file into event tokens.
        Handles edge cases: sparse files, tempo estimation failures, corrupted data.
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise ValueError(f"Failed to parse MIDI: {e}")

        # Handle sparse/empty files
        total_notes = sum(len(instr.notes) for instr in midi.instruments)
        if total_notes == 0:
            raise ValueError("MIDI file has no notes")
        if total_notes < 2:
            raise ValueError(f"MIDI file has only {total_notes} note(s), skipping (need ≥2)")

        # Estimate tempo with fallback
        try:
            tempo = midi.estimate_tempo()
        except (ValueError, RuntimeError):
            # Fallback: use default tempo or extract from tempo changes
            if midi.tempo_changes:
                tempo = midi.tempo_changes[0].tempo
            else:
                tempo = 120.0  # Default BPM

        # Clamp tempo to reasonable range
        tempo = max(40, min(240, tempo))
        tempo_bin = int((tempo - 40) / 10)  # Bin into 10 BPM buckets
        tempo_bin = max(0, min(19, tempo_bin))  # 0-19 bins

        # Extract notes from all instruments
        all_notes = []
        for instr in midi.instruments:
            if instr.is_drum:
                continue  # Skip drum tracks for now
            all_notes.extend(instr.notes)

        if not all_notes:
            raise ValueError("No non-drum notes found")

        # Collect all note events
        events = []
        instruments = [midi.instruments[0]] if self.melody_only and midi.instruments else midi.instruments
        
        for instrument in instruments:
            if instrument.is_drum:
                continue
            
            for note in instrument.notes:
                # Note on event
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': (note.end - note.start) * 1000,  # ms
                })
                # Note off event
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch,
                })
        
        # Sort by time
        events.sort(key=lambda x: x['time'])
        
        # Convert to tokens
        tokens = [self.vocab.BOS_TOKEN]
        current_time = 0.0
        
        for event in events:
            # Add time shift if needed
            time_diff = (event['time'] - current_time) * 1000  # ms
            if time_diff > 0:
                time_token = self.vocab.quantize_time_shift(time_diff, self.time_shift_ms)
                tokens.append(time_token)
            
            # Add event tokens
            if event['type'] == 'note_on':
                tokens.append(f"NOTE_ON_{event['pitch']}")
                tokens.append(self.vocab.quantize_velocity(event['velocity']))
                tokens.append(self.vocab.quantize_duration(event['duration'], self.max_duration_ms))
            else:
                tokens.append(f"NOTE_OFF_{event['pitch']}")
            
            current_time = event['time']
        
        tokens.append(self.vocab.EOS_TOKEN)
        
        metadata = {
            'tempo': tempo,
            'key': self._estimate_key(midi),
            'num_notes': len([e for e in events if e['type'] == 'note_on']),
            'duration': events[-1]['time'] if events else 0,
        }
        
        return tokens, metadata
    
    def _estimate_key(self, midi: pretty_midi.PrettyMIDI) -> str:
        """Estimate key signature (simple heuristic)."""
        # Count pitch classes
        pitch_counts = defaultdict(int)
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                pitch_counts[note.pitch % 12] += 1
        
        if not pitch_counts:
            return "C"
        
        # Simple major key detection (Krumhansl-Schmuckler would be better)
        major_profiles = {
            0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"
        }
        
        most_common = max(pitch_counts.items(), key=lambda x: x[1])[0]
        return major_profiles.get(most_common, "C")


def prepare_dataset(
    midi_dir: Path,
    output_path: Path,
    config: dict,
    vocab: MusicVocabulary,
) -> Dict:
    """Prepare dataset from MIDI files.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_path: Path to save processed dataset (.npz)
        config: Configuration dictionary
        vocab: Music vocabulary
        
    Returns:
        Statistics dictionary
    """
    data_config = config.get('data', {})
    tok_config = config.get('tokenization', {})
    
    tokenizer = MIDITokenizer(
        vocab=vocab,
        time_shift_ms=tok_config.get('time_shift_ms', 20),
        max_duration_ms=tok_config.get('max_duration_ms', 2000),
        melody_only=data_config.get('melody_only', False),
    )
    
    # Find all MIDI files
    midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")
    
    # Tokenize all files
    all_sequences = []
    all_metadata = []
    skipped = 0
    
    for midi_path in tqdm(midi_files, desc="Tokenizing MIDI"):
        try:
            tokens, metadata = tokenizer.tokenize_midi(midi_path)
        except Exception as e:
            # Skip corrupted/invalid files silently
            skipped += 1
            continue
        
        if not tokens:
            continue
        
        # Filter by length
        min_len = data_config.get('min_midi_length', 32)
        max_len = data_config.get('max_midi_length', 16384)
        
        if len(tokens) < min_len or len(tokens) > max_len:
            continue
        
        # Encode to IDs
        token_ids = vocab.encode_sequence(tokens)
        all_sequences.append(np.array(token_ids, dtype=np.int32))
        all_metadata.append(metadata)
    
    logger.info(f"Processed {len(all_sequences)} valid sequences (skipped {skipped} corrupted/invalid files)")
    
    if len(all_sequences) == 0:
        raise ValueError("No valid MIDI sequences to train on! Check your dataset.")
    
    # Stratified split by tempo/key buckets
    tempo_bins = [60, 90, 120, 150, 180]  # BPM buckets
    
    def get_bucket(metadata):
        tempo = metadata.get('tempo', 120)
        tempo_bucket = sum(1 for t in tempo_bins if tempo >= t)
        key_bucket = hash(metadata.get('key', 'C')) % 12
        return tempo_bucket * 12 + key_bucket
    
    buckets = [get_bucket(m) for m in all_metadata]
    
    # Split indices
    train_ratio = data_config.get('train_split', 0.8)
    val_ratio = data_config.get('val_split', 0.1)
    seed = data_config.get('seed', 42)
    
    indices = np.arange(len(all_sequences))
    
    # Train + temp split
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=buckets,
        random_state=seed,
    )
    
    # Val + test split
    temp_buckets = [buckets[i] for i in temp_idx]
    val_size = val_ratio / (1 - train_ratio)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=temp_buckets,
        random_state=seed,
    )
    
    # Create datasets
    train_seqs = [all_sequences[i] for i in train_idx]
    val_seqs = [all_sequences[i] for i in val_idx]
    test_seqs = [all_sequences[i] for i in test_idx]
    
    # Flatten and create chunks for training
    seq_len = data_config.get('sequence_length', 512)
    
    def chunk_sequences(sequences, seq_len):
        """Split sequences into fixed-length chunks."""
        chunks = []
        for seq in sequences:
            for i in range(0, len(seq) - seq_len, seq_len // 2):  # 50% overlap
                chunk = seq[i:i + seq_len]
                if len(chunk) == seq_len:
                    chunks.append(chunk)
        return np.array(chunks, dtype=np.int32)
    
    train_data = chunk_sequences(train_seqs, seq_len)
    val_data = chunk_sequences(val_seqs, seq_len)
    test_data = chunk_sequences(test_seqs, seq_len)
    
    logger.info(f"Train: {len(train_data)} chunks, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train=train_data,
        val=val_data,
        test=test_data,
        vocab_size=np.array([vocab.vocab_size]),
    )
    
    # Compute statistics
    all_tokens = np.concatenate([train_data.flatten(), val_data.flatten(), test_data.flatten()])
    unique, counts = np.unique(all_tokens, return_counts=True)
    token_freq = dict(zip(unique.tolist(), counts.tolist()))
    
    tempo_values = [m['tempo'] for m in all_metadata]
    key_values = [m['key'] for m in all_metadata]
    
    stats = {
        'num_files': len(all_sequences),
        'train_chunks': len(train_data),
        'val_chunks': len(val_data),
        'test_chunks': len(test_data),
        'vocab_size': vocab.vocab_size,
        'sequence_length': seq_len,
        'token_frequencies': token_freq,
        'tempo_stats': {
            'mean': float(np.mean(tempo_values)),
            'std': float(np.std(tempo_values)),
            'min': float(np.min(tempo_values)),
            'max': float(np.max(tempo_values)),
        },
        'key_distribution': dict(zip(*np.unique(key_values, return_counts=True))),
    }
    
    # Save stats
    stats_path = output_path.parent / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Statistics saved to {stats_path}")
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare MIDI dataset")
    parser.add_argument("--midi_dir", type=Path, required=True, help="Directory with MIDI files")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create vocabulary
    vocab = create_vocab_from_config(config)
    
    # Save vocabulary
    vocab_path = args.out.parent / "vocab.json"
    vocab.save(vocab_path)
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    # Prepare dataset
    prepare_dataset(args.midi_dir, args.out, config, vocab)


if __name__ == "__main__":
    main()
