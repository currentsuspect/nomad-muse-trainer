"""Muse Student Report Card - Comprehensive model evaluation.

Evaluates music generation models for:
- Perplexity and loss metrics
- Top-k accuracy 
- Memorization/plagiarism detection
- Distribution analysis
- Sample generation and analysis
"""

import argparse
import json
import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import pretty_midi

from .vocab import MusicVocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, onnx_path: Path, vocab: MusicVocabulary):
        """Initialize evaluator.
        
        Args:
            onnx_path: Path to ONNX model
            vocab: Music vocabulary
        """
        self.onnx_path = onnx_path
        self.vocab = vocab
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def compute_nll_loss(self, data: np.ndarray, batch_size: int = 32) -> Tuple[float, int]:
        """Compute negative log-likelihood loss.
        
        Args:
            data: Token sequences [num_seqs, seq_len]
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (average_nll_loss, total_tokens)
        """
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        total_loss = 0.0
        total_tokens = 0
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Computing NLL loss"):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            
            inputs = batch_data[:, :-1]
            targets = batch_data[:, 1:]
            
            # ONNX inference
            logits = self.session.run(
                [self.output_name],
                {self.input_name: inputs.astype(np.int64)}
            )[0]
            
            logits_tensor = torch.from_numpy(logits).float()
            targets_tensor = torch.from_numpy(targets).long()
            
            # Compute loss
            loss = criterion(
                logits_tensor.reshape(-1, self.vocab.vocab_size),
                targets_tensor.reshape(-1)
            )
            
            total_loss += loss.item()
            total_tokens += (targets != 0).sum()
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss, total_tokens
    
    def compute_perplexity(self, data: np.ndarray, batch_size: int = 32) -> float:
        """Compute perplexity.
        
        Args:
            data: Token sequences
            batch_size: Batch size
            
        Returns:
            Perplexity value
        """
        avg_loss, _ = self.compute_nll_loss(data, batch_size)
        return np.exp(avg_loss)
    
    def compute_top_k_accuracy(self, data: np.ndarray, k_values: List[int] = [1, 3, 5, 10], 
                              batch_size: int = 32) -> Dict[int, float]:
        """Compute top-k accuracy.
        
        Args:
            data: Token sequences
            k_values: List of k values
            batch_size: Batch size
            
        Returns:
            Dictionary mapping k to accuracy
        """
        correct_counts = {k: 0 for k in k_values}
        total_predictions = 0
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Computing top-k accuracy"):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            
            inputs = batch_data[:, :-1]
            targets = batch_data[:, 1:]
            
            # ONNX inference
            logits = self.session.run(
                [self.output_name],
                {self.input_name: inputs.astype(np.int64)}
            )[0]
            
            # Get top-k predictions
            max_k = max(k_values)
            top_k_indices = np.argsort(logits, axis=-1)[:, :, -max_k:][:, :, ::-1]
            
            # Check accuracy for each k
            for k in k_values:
                top_k = top_k_indices[:, :, :k]
                
                # Check if true target is in top-k
                targets_expanded = targets[:, :, np.newaxis]
                matches = (top_k == targets_expanded).any(axis=-1)
                
                # Only count non-padding tokens
                valid_mask = targets != 0
                correct_counts[k] += (matches & valid_mask).sum()
            
            total_predictions += (targets != 0).sum()
        
        accuracies = {k: count / max(total_predictions, 1) for k, count in correct_counts.items()}
        return accuracies
    
    def compute_token_entropy(self, data: np.ndarray, batch_size: int = 32) -> float:
        """Compute average token entropy.
        
        Args:
            data: Token sequences
            batch_size: Batch size
            
        Returns:
            Average entropy across all predictions
        """
        total_entropy = 0.0
        total_predictions = 0
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Computing token entropy"):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            
            inputs = batch_data[:, :-1]
            targets = batch_data[:, 1:]
            
            # ONNX inference
            logits = self.session.run(
                [self.output_name],
                {self.input_name: inputs.astype(np.int64)}
            )[0]
            
            # Convert to probabilities
            probs = torch.softmax(torch.from_numpy(logits), dim=-1)
            
            # Get probabilities for actual targets
            target_probs = probs.gather(-1, torch.from_numpy(targets).unsqueeze(-1)).squeeze(-1)
            
            # Only consider non-padding tokens
            valid_mask = targets != 0
            valid_probs = target_probs[valid_mask]
            
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(torch.log(valid_probs + 1e-8))
            
            total_entropy += entropy.item()
            total_predictions += valid_mask.sum()
        
        avg_entropy = total_entropy / max(total_predictions, 1)
        return avg_entropy


class MemorizationDetector:
    """Detect memorization using n-gram similarity."""
    
    def __init__(self, ngram_size: int = 4, max_train_chunks: int = 2000):
        """Initialize detector.
        
        Args:
            ngram_size: Size of n-grams for comparison
            max_train_chunks: Maximum training chunks to compare against
        """
        self.ngram_size = ngram_size
        self.max_train_chunks = max_train_chunks
        self.train_chunks = []
        
    def extract_ngrams(self, sequence: List[int]) -> set:
        """Extract n-grams from sequence.
        
        Args:
            sequence: Token sequence
            
        Returns:
            Set of n-grams
        """
        if len(sequence) < self.ngram_size:
            return set()
        
        ngrams = set()
        for i in range(len(sequence) - self.ngram_size + 1):
            ngram = tuple(sequence[i:i + self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def load_training_chunks(self, train_data: np.ndarray):
        """Load training chunks for comparison.
        
        Args:
            train_data: Training sequences
        """
        logger.info(f"Loading {min(len(train_data), self.max_train_chunks)} training chunks...")
        
        # Sample random chunks if dataset is large
        indices = np.random.choice(
            len(train_data), 
            size=min(len(train_data), self.max_train_chunks), 
            replace=False
        )
        
        self.train_chunks = []
        for idx in indices:
            sequence = train_data[idx].tolist()
            # Remove padding
            sequence = [x for x in sequence if x != 0]
            if len(sequence) >= self.ngram_size:
                ngrams = self.extract_ngrams(sequence)
                self.train_chunks.append(ngrams)
        
        logger.info(f"Loaded {len(self.train_chunks)} training chunks")
    
    def detect_memorization(self, generated_sequence: List[int]) -> Dict[str, Any]:
        """Detect memorization in generated sequence.
        
        Args:
            generated_sequence: Generated token sequence
            
        Returns:
            Detection results
        """
        if not self.train_chunks:
            return {"error": "No training chunks loaded"}
        
        # Extract n-grams from generated sequence
        generated_ngrams = self.extract_ngrams(generated_sequence)
        
        if not generated_ngrams:
            return {
                "max_similarity": 0.0,
                "mean_top10_similarity": 0.0,
                "closest_chunks": [],
                "interpretation": "too_short"
            }
        
        # Compute similarities with training chunks
        similarities = []
        for i, train_ngrams in enumerate(self.train_chunks):
            similarity = self.jaccard_similarity(generated_ngrams, train_ngrams)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Get statistics
        max_similarity = similarities[0][0] if similarities else 0.0
        top_10_similarities = [s[0] for s in similarities[:10]]
        mean_top10_similarity = np.mean(top_10_similarities) if top_10_similarities else 0.0
        
        # Get closest chunks
        closest_chunks = [
            {
                "similarity": sim,
                "chunk_index": idx,
                "position": "unknown"  # We don't track positions for sampled chunks
            }
            for sim, idx in similarities[:5]
        ]
        
        # Interpretation
        if max_similarity < 0.15:
            interpretation = "likely_novel"
        elif max_similarity < 0.30:
            interpretation = "borrowing_structure"
        else:
            interpretation = "suspicious_memorization"
        
        return {
            "max_similarity": max_similarity,
            "mean_top10_similarity": mean_top10_similarity,
            "closest_chunks": closest_chunks,
            "interpretation": interpretation,
            "ngram_size": self.ngram_size,
            "num_generated_ngrams": len(generated_ngrams),
            "num_train_chunks": len(self.train_chunks)
        }


class DistributionAnalyzer:
    """Analyze token distribution for sanity checks."""
    
    def __init__(self, vocab: MusicVocabulary):
        """Initialize analyzer.
        
        Args:
            vocab: Music vocabulary
        """
        self.vocab = vocab
        
    def analyze_token_distribution(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """Analyze token distribution in sequences.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Distribution analysis results
        """
        # Flatten all tokens
        all_tokens = []
        for seq in sequences:
            all_tokens.extend([t for t in seq if t != 0])  # Remove padding
        
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Categorize tokens
        note_on_count = sum(count for token_id, count in token_counts.items() 
                           if self.vocab.decode(token_id).startswith("NOTE_ON_"))
        note_off_count = sum(count for token_id, count in token_counts.items() 
                           if self.vocab.decode(token_id).startswith("NOTE_OFF_"))
        time_shift_count = sum(count for token_id, count in token_counts.items() 
                             if self.vocab.decode(token_id).startswith("TIME_SHIFT_"))
        velocity_count = sum(count for token_id, count in token_counts.items() 
                           if self.vocab.decode(token_id).startswith("VELOCITY_"))
        duration_count = sum(count for token_id, count in token_counts.items() 
                           if self.vocab.decode(token_id).startswith("DURATION_"))
        
        # Calculate ratios
        ratios = {
            "note_on_ratio": note_on_count / total_tokens if total_tokens > 0 else 0,
            "note_off_ratio": note_off_count / total_tokens if total_tokens > 0 else 0,
            "time_shift_ratio": time_shift_count / total_tokens if total_tokens > 0 else 0,
            "velocity_ratio": velocity_count / total_tokens if total_tokens > 0 else 0,
            "duration_ratio": duration_count / total_tokens if total_tokens > 0 else 0,
        }
        
        # Velocity distribution
        velocity_bins = {}
        for token_id, count in token_counts.items():
            token_str = self.vocab.decode(token_id)
            if token_str.startswith("VELOCITY_"):
                bin_num = int(token_str.split("_")[-1])
                velocity_bins[bin_num] = velocity_bins.get(bin_num, 0) + count
        
        # Duration distribution  
        duration_bins = {}
        for token_id, count in token_counts.items():
            token_str = self.vocab.decode(token_id)
            if token_str.startswith("DURATION_"):
                bin_num = int(token_str.split("_")[-1])
                duration_bins[bin_num] = duration_bins.get(bin_num, 0) + count
        
        return {
            "total_tokens": total_tokens,
            "unique_tokens": len(token_counts),
            "ratios": ratios,
            "velocity_distribution": velocity_bins,
            "duration_distribution": duration_bins,
            "most_common_tokens": dict(token_counts.most_common(20))
        }
    
    def compute_loopiness_score(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """Compute "loopiness" score based on repeated patterns.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Loopiness analysis
        """
        all_sequences = []
        for seq in sequences:
            # Remove padding and convert to tuples for easier processing
            clean_seq = tuple(t for t in seq if t != 0)
            if len(clean_seq) > 0:
                all_sequences.append(clean_seq)
        
        # Find repeated bigrams
        bigram_counts = Counter()
        for seq in all_sequences:
            for i in range(len(seq) - 1):
                bigram = (seq[i], seq[i + 1])
                bigram_counts[bigram] += 1
        
        repeated_bigram_rate = sum(count for count in bigram_counts.values() if count > 1) / len(bigram_counts) if len(bigram_counts) > 0 else 0.0
        
        # Find longest repeated 8-gram streak
        eightgram_counts = Counter()
        for seq in all_sequences:
            for i in range(len(seq) - 7):
                eightgram = seq[i:i + 8]
                eightgram_counts[eightgram] += 1
        
        max_repeated_eightgram = max(eightgram_counts.values()) if eightgram_counts else 1
        
        return {
            "repeated_bigram_rate": repeated_bigram_rate,
            "max_repeated_eightgram_count": max_repeated_eightgram,
            "total_unique_bigrams": len(bigram_counts),
            "total_unique_8grams": len(eightgram_counts)
        }


class SampleGenerator:
    """Generate and analyze samples."""
    
    def __init__(self, evaluator: ModelEvaluator, vocab: MusicVocabulary):
        """Initialize generator.
        
        Args:
            evaluator: Model evaluator instance
            vocab: Music vocabulary
        """
        self.evaluator = evaluator
        self.vocab = vocab
        
    def generate_sample(self, primer: List[int], max_length: int = 512, 
                       temperature: float = 1.0) -> List[int]:
        """Generate a sample sequence.
        
        Args:
            primer: Initial token sequence
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        sequence = list(primer)
        
        for _ in range(max_length):
            # Prepare input (last 512 tokens)
            context = sequence[-512:] if len(sequence) > 512 else sequence
            input_array = np.array([context], dtype=np.int64)
            
            # Run inference
            logits = self.evaluator.session.run(
                [self.evaluator.output_name],
                {self.evaluator.input_name: input_array}
            )[0]
            
            # Get logits for last position
            next_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Softmax and sample
            exp_logits = np.exp(next_logits - np.max(next_logits))
            probs = exp_logits / exp_logits.sum()
            next_token = np.random.choice(len(probs), p=probs)
            
            sequence.append(int(next_token))
            
            # Stop at EOS
            if next_token == self.vocab.eos_id:
                break
        
        return sequence
    
    def tokens_to_midi(self, tokens: List[int], output_path: Path, tempo: int = 120):
        """Convert token sequence to MIDI file.
        
        Args:
            tokens: Token IDs
            output_path: Output MIDI path
            tempo: Tempo in BPM
        """
        # Convert to strings
        token_strings = self.vocab.decode_sequence(tokens)
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        current_time = 0.0
        active_notes = {}  # pitch -> start_time
        
        for token in token_strings:
            if token.startswith("TIME_SHIFT_"):
                # Advance time
                bins = int(token.split("_")[-1])
                current_time += bins * 0.02  # 20ms per bin
                
            elif token.startswith("NOTE_ON_"):
                pitch = int(token.split("_")[-1])
                active_notes[pitch] = current_time
                
            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.split("_")[-1])
                if pitch in active_notes:
                    start = active_notes.pop(pitch)
                    note = pretty_midi.Note(
                        velocity=64,
                        pitch=pitch,
                        start=start,
                        end=current_time,
                    )
                    instrument.notes.append(note)
        
        # Close any remaining active notes
        for pitch, start in active_notes.items():
            note = pretty_midi.Note(
                velocity=64,
                pitch=pitch,
                start=start,
                end=current_time + 0.5,
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        midi.write(str(output_path))
    
    def analyze_midi(self, midi_path: Path) -> Dict[str, Any]:
        """Analyze generated MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            MIDI analysis results
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        instrument = midi.instruments[0] if midi.instruments else None
        
        if not instrument:
            return {
                "duration": 0.0,
                "note_count": 0,
                "notes_per_second": 0.0,
                "pitch_range": [0, 127],
                "unique_pitches": 0
            }
        
        notes = instrument.notes
        duration = midi.get_end_time()
        
        # Compute metrics
        note_count = len(notes)
        notes_per_second = note_count / duration if duration > 0 else 0
        
        if notes:
            pitches = [note.pitch for note in notes]
            pitch_range = [min(pitches), max(pitches)]
            unique_pitches = len(set(pitches))
        else:
            pitch_range = [0, 0]
            unique_pitches = 0
        
        return {
            "duration": duration,
            "note_count": note_count,
            "notes_per_second": notes_per_second,
            "pitch_range": pitch_range,
            "unique_pitches": unique_pitches
        }


def generate_sample_manifest(samples_dir: Path) -> Dict[str, Any]:
    """Generate manifest for all samples in directory.
    
    Args:
        samples_dir: Directory containing MIDI samples
        
    Returns:
        Sample manifest
    """
    manifest = {
        "samples": [],
        "summary": {}
    }
    
    midi_files = list(samples_dir.glob("*.mid"))
    
    for midi_file in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_file))
            instrument = midi.instruments[0] if midi.instruments else None
            
            if instrument:
                notes = instrument.notes
                duration = midi.get_end_time()
                note_count = len(notes)
                notes_per_second = note_count / duration if duration > 0 else 0
                
                if notes:
                    pitches = [note.pitch for note in notes]
                    pitch_range = [min(pitches), max(pitches)]
                    unique_pitches = len(set(pitches))
                else:
                    pitch_range = [0, 0]
                    unique_pitches = 0
                
                sample_info = {
                    "filename": midi_file.name,
                    "duration": duration,
                    "note_count": note_count,
                    "notes_per_second": notes_per_second,
                    "pitch_range": pitch_range,
                    "unique_pitches": unique_pitches
                }
                
                manifest["samples"].append(sample_info)
                
        except Exception as e:
            logger.warning(f"Failed to analyze {midi_file}: {e}")
    
    # Generate summary statistics
    if manifest["samples"]:
        durations = [s["duration"] for s in manifest["samples"]]
        note_counts = [s["note_count"] for s in manifest["samples"]]
        notes_per_second = [s["notes_per_second"] for s in manifest["samples"]]
        
        manifest["summary"] = {
            "total_samples": len(manifest["samples"]),
            "avg_duration": np.mean(durations),
            "avg_note_count": np.mean(note_counts),
            "avg_notes_per_second": np.mean(notes_per_second),
            "total_notes": sum(note_counts)
        }
    
    return manifest


def main():
    """Main entry point for report card generation."""
    parser = argparse.ArgumentParser(description="Generate Muse Student Report Card")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.npz")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--out", type=Path, required=True, help="Output report JSON path")
    parser.add_argument("--samples_dir", type=Path, help="Directory for generated samples")
    parser.add_argument("--num_samples", type=int, default=12, help="Number of samples to generate")
    parser.add_argument("--length", type=int, default=512, help="Sample length in tokens")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6, 0.8, 1.0], 
                       help="Sampling temperatures")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4], 
                       help="Random seeds for generation")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--primer_length", type=int, default=16, help="Primer length in tokens")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load vocabulary
    vocab_path = args.onnx.parent / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path("artifacts/vocab.json")
    
    vocab = MusicVocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary: {vocab.vocab_size} tokens")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    data = np.load(args.dataset)
    train_data = data['train']
    val_data = data.get('val', data['test'])  # fallback to test if val doesn't exist
    test_data = data.get('test', val_data)
    vocab_size = int(data['vocab_size'][0])
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} sequences")
    
    # Initialize evaluator
    logger.info(f"Loading ONNX model from {args.onnx}")
    evaluator = ModelEvaluator(args.onnx, vocab)
    
    # Initialize components
    memorization_detector = MemorizationDetector()
    memorization_detector.load_training_chunks(train_data)
    
    distribution_analyzer = DistributionAnalyzer(vocab)
    sample_generator = SampleGenerator(evaluator, vocab)
    
    # Initialize report
    report = {
        "metadata": {
            "dataset_path": str(args.dataset),
            "model_path": str(args.onnx),
            "vocab_size": vocab.vocab_size,
            "generation_config": {
                "num_samples": args.num_samples,
                "sample_length": args.length,
                "temperatures": args.temperatures,
                "seeds": args.seeds,
                "primer_length": args.primer_length
            }
        },
        "metrics": {},
        "memorization": {},
        "distribution": {},
        "samples": {}
    }
    
    # Compute validation metrics
    logger.info("Computing validation metrics...")
    val_nll, val_tokens = evaluator.compute_nll_loss(val_data)
    val_perplexity = evaluator.compute_perplexity(val_data)
    val_top_k = evaluator.compute_top_k_accuracy(val_data)
    val_entropy = evaluator.compute_token_entropy(val_data)
    
    report["metrics"]["validation"] = {
        "nll_loss": float(val_nll),
        "perplexity": float(val_perplexity),
        "top_k_accuracy": {k: float(v) for k, v in val_top_k.items()},
        "token_entropy": float(val_entropy),
        "total_tokens": int(val_tokens)
    }
    
    # Compute test metrics if available
    if len(test_data) > 0:
        logger.info("Computing test metrics...")
        test_nll, test_tokens = evaluator.compute_nll_loss(test_data)
        test_perplexity = evaluator.compute_perplexity(test_data)
        test_top_k = evaluator.compute_top_k_accuracy(test_data)
        test_entropy = evaluator.compute_token_entropy(test_data)
        
        report["metrics"]["test"] = {
            "nll_loss": float(test_nll),
            "perplexity": float(test_perplexity),
            "top_k_accuracy": {k: float(v) for k, v in test_top_k.items()},
            "token_entropy": float(test_entropy),
            "total_tokens": int(test_tokens)
        }
    
    # Analyze training distribution for comparison
    logger.info("Analyzing training distribution...")
    train_sequences = [seq.tolist() for seq in train_data[:1000]]  # Sample for efficiency
    train_distribution = distribution_analyzer.analyze_token_distribution(train_sequences)
    train_loopiness = distribution_analyzer.compute_loopiness_score(train_sequences)
    
    report["distribution"]["training"] = {
        "token_distribution": train_distribution,
        "loopiness": train_loopiness
    }
    
    # Generate samples
    if args.samples_dir:
        logger.info(f"Generating {args.num_samples} samples...")
        samples_dir = args.samples_dir
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Use first sequence as primer
        primer = train_data[0, :args.primer_length].tolist()
        
        sample_manifest = {
            "samples": [],
            "config": {
                "temperatures": args.temperatures,
                "seeds": args.seeds,
                "primer_length": args.primer_length,
                "sample_length": args.length
            }
        }
        
        sample_idx = 0
        for temp in args.temperatures:
            for seed in args.seeds:
                if sample_idx >= args.num_samples:
                    break
                
                # Set random seed
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Generate sample
                generated = sample_generator.generate_sample(
                    primer, args.length, temp
                )
                
                # Save MIDI
                filename = f"sample_t{temp}_seed{seed}.mid"
                midi_path = samples_dir / filename
                sample_generator.tokens_to_midi(generated, midi_path)
                
                # Analyze MIDI
                analysis = sample_generator.analyze_midi(midi_path)
                analysis.update({
                    "temperature": temp,
                    "seed": seed,
                    "filename": filename,
                    "token_length": len(generated)
                })
                
                sample_manifest["samples"].append(analysis)
                sample_idx += 1
        
        # Run memorization detection on samples
        logger.info("Running memorization detection...")
        sample_sequences = []
        for temp in args.temperatures:
            for seed in args.seeds:
                if sample_idx >= args.num_samples:
                    break
                
                # Set seed and regenerate for consistency
                np.random.seed(seed)
                torch.manual_seed(seed)
                generated = sample_generator.generate_sample(primer, args.length, temp)
                sample_sequences.append(generated)
        
        # Detect memorization in generated samples
        memorization_results = []
        for i, seq in enumerate(sample_sequences[:min(len(sample_sequences), 10)]):  # Limit for speed
            result = memorization_detector.detect_memorization(seq)
            result["sample_index"] = i
            # Convert numpy types to Python types
            if "max_similarity" in result:
                result["max_similarity"] = float(result["max_similarity"])
            if "mean_top10_similarity" in result:
                result["mean_top10_similarity"] = float(result["mean_top10_similarity"])
            memorization_results.append(result)
        
        report["memorization"]["generated_samples"] = memorization_results
        
        # Analyze sample distribution
        logger.info("Analyzing sample distribution...")
        sample_tokens = []
        for seq in sample_sequences:
            sample_tokens.extend([t for t in seq if t != 0])
        
        sample_distribution = distribution_analyzer.analyze_token_distribution([sample_tokens])
        sample_loopiness = distribution_analyzer.compute_loopiness_score(sample_sequences)
        
        report["distribution"]["generated_samples"] = {
            "token_distribution": sample_distribution,
            "loopiness": sample_loopiness
        }
        
        # Save sample manifest
        manifest_path = samples_dir / "samples_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(sample_manifest, f, indent=2)
        
        report["samples"] = {
            "manifest_path": str(manifest_path),
            "summary": json.loads(json.dumps(generate_sample_manifest(samples_dir), default=str))
        }
    
    # Save report
    logger.info(f"Saving report to {args.out}")
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Report card generation complete!")
    
    # Print summary
    print("\n" + "="*70)
    print("MUSE STUDENT REPORT CARD SUMMARY")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.onnx}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    print()
    
    if "validation" in report["metrics"]:
        val_metrics = report["metrics"]["validation"]
        print(f"Validation Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"Validation NLL Loss: {val_metrics['nll_loss']:.4f}")
        print(f"Token Entropy: {val_metrics['token_entropy']:.2f}")
        print()
        
        print("Top-k Accuracy:")
        for k, acc in val_metrics['top_k_accuracy'].items():
            print(f"  Top-{k}: {acc:.4f}")
        print()
    
    if "memorization" in report and "generated_samples" in report["memorization"]:
        mem_results = report["memorization"]["generated_samples"]
        if mem_results:
            max_similarities = [r["max_similarity"] for r in mem_results if "max_similarity" in r]
            if max_similarities:
                avg_max_sim = np.mean(max_similarities)
                print(f"Average Max Similarity: {avg_max_sim:.3f}")
                interpretations = [r["interpretation"] for r in mem_results]
                print(f"Most common interpretation: {Counter(interpretations).most_common(1)[0][0]}")
    
    print("="*70)


if __name__ == "__main__":
    main()