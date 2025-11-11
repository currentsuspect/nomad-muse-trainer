"""Evaluation: perplexity, top-k accuracy, and sample generation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

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


def compute_perplexity(
    model_or_session,
    data: np.ndarray,
    vocab_size: int,
    is_onnx: bool = False,
    batch_size: int = 32,
) -> float:
    """Compute perplexity on dataset.
    
    Args:
        model_or_session: PyTorch model or ONNX session
        data: Token sequences [num_seqs, seq_len]
        vocab_size: Vocabulary size
        is_onnx: Whether using ONNX model
        batch_size: Batch size for evaluation
        
    Returns:
        Perplexity value
    """
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Computing perplexity"):
        batch_data = data[i * batch_size:(i + 1) * batch_size]
        
        inputs = batch_data[:, :-1]
        targets = batch_data[:, 1:]
        
        if is_onnx:
            # ONNX inference
            input_name = model_or_session.get_inputs()[0].name
            output_name = model_or_session.get_outputs()[0].name
            
            logits = model_or_session.run(
                [output_name],
                {input_name: inputs.astype(np.int64)}
            )[0]
            
            logits_tensor = torch.from_numpy(logits).float()
            targets_tensor = torch.from_numpy(targets).long()
        else:
            # PyTorch inference
            model_or_session.eval()
            with torch.no_grad():
                inputs_tensor = torch.from_numpy(inputs).long()
                targets_tensor = torch.from_numpy(targets).long()
                
                if hasattr(model_or_session, 'gru'):
                    logits_tensor, _ = model_or_session(inputs_tensor)
                else:
                    logits_tensor = model_or_session(inputs_tensor)
        
        # Compute loss
        loss = criterion(
            logits_tensor.reshape(-1, vocab_size),
            targets_tensor.reshape(-1)
        )
        
        total_loss += loss.item()
        total_tokens += (targets != 0).sum()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def compute_top_k_accuracy(
    model_or_session,
    data: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
    is_onnx: bool = False,
    batch_size: int = 32,
) -> Dict[int, float]:
    """Compute top-k accuracy.
    
    Args:
        model_or_session: PyTorch model or ONNX session
        data: Token sequences
        k_values: List of k values to compute
        is_onnx: Whether using ONNX model
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
        
        if is_onnx:
            input_name = model_or_session.get_inputs()[0].name
            output_name = model_or_session.get_outputs()[0].name
            
            logits = model_or_session.run(
                [output_name],
                {input_name: inputs.astype(np.int64)}
            )[0]
        else:
            model_or_session.eval()
            with torch.no_grad():
                inputs_tensor = torch.from_numpy(inputs).long()
                
                if hasattr(model_or_session, 'gru'):
                    logits_tensor, _ = model_or_session(inputs_tensor)
                else:
                    logits_tensor = model_or_session(inputs_tensor)
                
                logits = logits_tensor.numpy()
        
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


def sample_from_onnx(
    session: ort.InferenceSession,
    vocab: MusicVocabulary,
    primer: List[int],
    max_length: int = 256,
    temperature: float = 1.0,
) -> List[int]:
    """Generate sample using ONNX model.
    
    Args:
        session: ONNX inference session
        vocab: Music vocabulary
        primer: Initial token sequence
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated token sequence
    """
    sequence = list(primer)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for _ in range(max_length):
        # Prepare input (last tokens up to model's context)
        context = sequence[-512:] if len(sequence) > 512 else sequence
        input_array = np.array([context], dtype=np.int64)
        
        # Run inference
        logits = session.run([output_name], {input_name: input_array})[0]
        
        # Get logits for last position
        next_logits = logits[0, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            next_logits = next_logits / temperature
        
        # Softmax
        exp_logits = np.exp(next_logits - np.max(next_logits))
        probs = exp_logits / exp_logits.sum()
        
        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        sequence.append(int(next_token))
        
        # Stop at EOS
        if next_token == vocab.eos_id:
            break
    
    return sequence


def tokens_to_midi(
    tokens: List[str],
    vocab: MusicVocabulary,
    output_path: Path,
    tempo: int = 120,
):
    """Convert token sequence to MIDI file.
    
    Args:
        tokens: List of token strings
        vocab: Music vocabulary
        output_path: Path to save MIDI
        tempo: Tempo in BPM
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    
    current_time = 0.0
    active_notes = {}  # pitch -> start_time
    
    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            # Advance time
            bins = int(token.split("_")[-1])
            current_time += bins * 0.02  # 20ms per bin (from config)
        
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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.npz")
    parser.add_argument("--onnx", type=Path, help="Path to ONNX model")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--sample", action='store_true', help="Generate sample MIDI")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load vocabulary
    vocab_path = args.onnx.parent / "vocab.json" if args.onnx else Path("artifacts/vocab.json")
    vocab = MusicVocabulary.load(vocab_path)
    logger.info(f"Loaded vocabulary: {vocab.vocab_size} tokens")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    data = np.load(args.dataset)
    test_data = data['test']
    vocab_size = int(data['vocab_size'][0])
    
    logger.info(f"Test set: {len(test_data)} sequences")
    
    # Load model
    if args.onnx:
        logger.info(f"Loading ONNX model from {args.onnx}")
        session = ort.InferenceSession(str(args.onnx))
        is_onnx = True
        model = session
    else:
        raise ValueError("Please provide --onnx path")
    
    # Compute perplexity
    logger.info("Computing perplexity...")
    perplexity = compute_perplexity(model, test_data, vocab_size, is_onnx=is_onnx)
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    # Compute top-k accuracy
    eval_config = config.get('evaluation', {})
    k_values = eval_config.get('top_k_values', [1, 3, 5, 10])
    
    logger.info("Computing top-k accuracy...")
    accuracies = compute_top_k_accuracy(model, test_data, k_values, is_onnx=is_onnx)
    
    for k, acc in accuracies.items():
        logger.info(f"Top-{k} accuracy: {acc:.4f}")
    
    # Generate sample if requested
    if args.sample and is_onnx:
        logger.info("Generating sample MIDI...")
        
        # Use first few tokens from test set as primer
        primer_len = eval_config.get('sample_primer_length', 16)
        primer = test_data[0, :primer_len].tolist()
        
        sample_len = eval_config.get('sample_length', 256)
        temperature = eval_config.get('sample_temperature', 1.0)
        
        generated_ids = sample_from_onnx(session, vocab, primer, sample_len, temperature)
        generated_tokens = vocab.decode_sequence(generated_ids)
        
        # Save to MIDI
        output_path = args.onnx.parent / "sample_eval.mid"
        tokens_to_midi(generated_tokens, vocab, output_path)
        logger.info(f"Sample saved to {output_path}")


if __name__ == "__main__":
    main()
