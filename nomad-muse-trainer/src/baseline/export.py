"""Export baseline models to binary format.

Serializes Markov and rhythm models for lightweight inference.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

from .markov import VariableOrderMarkov
from .rhythm import RhythmHistogram, CombinedBaseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_baseline_models(
    sequences: list,
    config: dict
) -> CombinedBaseline:
    """Build baseline models from sequences.
    
    Args:
        sequences: List of token ID sequences
        config: Configuration dictionary
        
    Returns:
        Combined baseline model
    """
    baseline_config = config.get('baseline', {})
    
    # Train Markov model
    logger.info("Training Markov model...")
    markov_order = baseline_config.get('markov_order', 4)
    markov = VariableOrderMarkov(
        max_order=markov_order,
        smoothing=baseline_config.get('smoothing', 'witten-bell')
    )
    markov.train(sequences)
    logger.info(f"Markov model trained (order {markov_order}, vocab size {len(markov.vocab)})")
    
    # Train rhythm model
    logger.info("Training rhythm model...")
    rhythm = RhythmHistogram(bins_per_bar=16)
    rhythm.train(sequences)
    logger.info("Rhythm model trained")
    
    # Combine
    combined = CombinedBaseline(markov, rhythm, markov_weight=0.7)
    
    return combined


def export_baseline(
    model: CombinedBaseline,
    output_path: Path
):
    """Export baseline model to binary file.
    
    Args:
        model: Combined baseline model
        output_path: Path to save binary file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize with pickle
    with open(output_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Baseline model exported to {output_path}")
    
    # Compute size
    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info(f"Model size: {size_mb:.2f} MB")


def load_baseline(model_path: Path) -> CombinedBaseline:
    """Load baseline model from binary file.
    
    Args:
        model_path: Path to binary file
        
    Returns:
        Combined baseline model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build and export baseline model")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.npz")
    parser.add_argument("--out", type=Path, required=True, help="Output binary file")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    data = np.load(args.dataset)
    
    # Convert to list of sequences
    train_data = data['train']
    sequences = [seq.tolist() for seq in train_data]
    
    logger.info(f"Training on {len(sequences)} sequences")
    
    # Build baseline
    baseline = build_baseline_models(sequences, config)
    
    # Export
    export_baseline(baseline, args.out)


if __name__ == "__main__":
    main()
