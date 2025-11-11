"""Demo sample generator using ONNX runtime.

Generates a short MIDI sample from a priming sequence using the exported ONNX model.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pretty_midi
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vocab import MusicVocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_from_onnx(
    session: ort.InferenceSession,
    vocab: MusicVocabulary,
    primer: list,
    max_length: int = 256,
    temperature: float = 1.0,
) -> list:
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
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    logger.info(f"Generating {max_length} tokens (temperature={temperature})...")
    
    for i in range(max_length):
        # Prepare input (last tokens up to model's context)
        context = sequence[-512:] if len(sequence) > 512 else sequence
        input_array = np.array([context], dtype=np.int64)
        
        # Run inference
        try:
            logits = session.run([output_name], {input_name: input_array})[0]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            break
        
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
            logger.info(f"Generated {len(sequence)} tokens (stopped at EOS)")
            break
    
    return sequence


def tokens_to_midi(
    tokens: list,
    vocab: MusicVocabulary,
    output_path: Path,
    tempo: int = 120,
    time_shift_ms: float = 20.0,
):
    """Convert token sequence to MIDI file.
    
    Args:
        tokens: List of token strings
        vocab: Music vocabulary
        output_path: Path to save MIDI
        tempo: Tempo in BPM
        time_shift_ms: Milliseconds per time shift unit
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    
    current_time = 0.0
    active_notes = {}  # pitch -> (start_time, velocity)
    
    logger.info(f"Converting {len(tokens)} tokens to MIDI...")
    
    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            # Advance time
            try:
                bins = int(token.split("_")[-1])
                current_time += bins * (time_shift_ms / 1000.0)  # Convert to seconds
            except:
                pass
        
        elif token.startswith("NOTE_ON_"):
            try:
                pitch = int(token.split("_")[-1])
                if 0 <= pitch <= 127:
                    active_notes[pitch] = (current_time, 64)  # Default velocity
            except:
                pass
        
        elif token.startswith("NOTE_OFF_"):
            try:
                pitch = int(token.split("_")[-1])
                if pitch in active_notes:
                    start, velocity = active_notes.pop(pitch)
                    if current_time > start:  # Ensure positive duration
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=current_time,
                        )
                        instrument.notes.append(note)
            except:
                pass
        
        elif token.startswith("VELOCITY_"):
            # Update velocity for next note
            # (Simplified: just use default for now)
            pass
    
    # Close any remaining active notes
    for pitch, (start, velocity) in active_notes.items():
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=current_time + 0.5,
        )
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    
    logger.info(f"MIDI file saved to {output_path}")
    logger.info(f"Duration: {midi.get_end_time():.2f}s, Notes: {len(instrument.notes)}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate sample MIDI from ONNX model")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--out", type=Path, required=True, help="Output MIDI file")
    parser.add_argument("--vocab", type=Path, help="Path to vocab.json (default: same dir as onnx)")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--length", type=int, default=256, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--tempo", type=int, default=120, help="MIDI tempo (BPM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load config
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Load vocabulary
    vocab_path = args.vocab if args.vocab else args.onnx.parent / "vocab.json"
    
    if not vocab_path.exists():
        logger.error(f"Vocabulary file not found: {vocab_path}")
        return
    
    logger.info(f"Loading vocabulary from {vocab_path}")
    vocab = MusicVocabulary.load(vocab_path)
    logger.info(f"Vocabulary size: {vocab.vocab_size}")
    
    # Load ONNX model
    logger.info(f"Loading ONNX model from {args.onnx}")
    session = ort.InferenceSession(str(args.onnx))
    
    # Create primer sequence
    # Simple primer: BOS + a few notes
    primer = [
        vocab.bos_id,
        vocab.encode("TIME_SHIFT_4"),
        vocab.encode("NOTE_ON_60"),  # Middle C
        vocab.encode("VELOCITY_4"),
        vocab.encode("DURATION_8"),
        vocab.encode("TIME_SHIFT_8"),
        vocab.encode("NOTE_ON_64"),  # E
        vocab.encode("VELOCITY_4"),
        vocab.encode("DURATION_8"),
    ]
    
    logger.info(f"Primer: {len(primer)} tokens")
    
    # Generate
    generated_ids = sample_from_onnx(
        session,
        vocab,
        primer,
        max_length=args.length,
        temperature=args.temperature,
    )
    
    # Decode to tokens
    generated_tokens = vocab.decode_sequence(generated_ids)
    
    # Convert to MIDI
    time_shift_ms = config.get('tokenization', {}).get('time_shift_ms', 20.0)
    
    tokens_to_midi(
        generated_tokens,
        vocab,
        args.out,
        tempo=args.tempo,
        time_shift_ms=time_shift_ms,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
