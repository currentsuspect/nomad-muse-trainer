"""Rhythm baseline using bar-position histograms.

Models rhythm patterns based on position within musical bars.
"""

from collections import defaultdict
from typing import Dict, List
import numpy as np


class RhythmHistogram:
    """Histogram-based rhythm model."""
    
    def __init__(self, bins_per_bar: int = 16):
        """Initialize rhythm model.
        
        Args:
            bins_per_bar: Number of discrete positions per bar
        """
        self.bins_per_bar = bins_per_bar
        
        # Histograms: position -> {token: count}
        self.position_histograms: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Track total counts per position
        self.position_totals: Dict[int, int] = defaultdict(int)
    
    def train(self, sequences: List[List[int]], time_shift_token_prefix: str = "TIME_SHIFT"):
        """Train on token sequences.
        
        Args:
            sequences: List of token ID sequences
            time_shift_token_prefix: Prefix for time shift tokens
        """
        for seq in sequences:
            position = 0  # Current position in bar (in bins)
            
            for token in seq:
                # Update histogram
                bin_pos = position % self.bins_per_bar
                self.position_histograms[bin_pos][token] += 1
                self.position_totals[bin_pos] += 1
                
                # Advance position based on time shifts
                # (This is simplified; in practice we'd decode the token)
                position += 1
    
    def predict_proba(self, current_position: int) -> Dict[int, float]:
        """Predict probability distribution based on bar position.
        
        Args:
            current_position: Current position in bar
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        bin_pos = current_position % self.bins_per_bar
        
        counts = self.position_histograms[bin_pos]
        total = self.position_totals[bin_pos]
        
        if total == 0:
            return {}
        
        # Normalize counts to probabilities
        probs = {token: count / total for token, count in counts.items()}
        
        return probs
    
    def sample(self, current_position: int, temperature: float = 1.0) -> int:
        """Sample token based on rhythm model.
        
        Args:
            current_position: Current position in bar
            temperature: Sampling temperature
            
        Returns:
            Sampled token ID
        """
        probs = self.predict_proba(current_position)
        
        if not probs:
            return 0
        
        tokens = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # Apply temperature
        if temperature != 1.0:
            probabilities = probabilities ** (1.0 / temperature)
            probabilities /= probabilities.sum()
        
        return np.random.choice(tokens, p=probabilities)


class CombinedBaseline:
    """Combined Markov + Rhythm baseline."""
    
    def __init__(
        self,
        markov_model,
        rhythm_model,
        markov_weight: float = 0.7,
    ):
        """Initialize combined model.
        
        Args:
            markov_model: Variable-order Markov model
            rhythm_model: Rhythm histogram model
            markov_weight: Weight for Markov model (1 - weight for rhythm)
        """
        self.markov = markov_model
        self.rhythm = rhythm_model
        self.markov_weight = markov_weight
    
    def predict_proba(self, context: List[int], position: int) -> Dict[int, float]:
        """Combined prediction.
        
        Args:
            context: Previous tokens
            position: Current position in bar
            
        Returns:
            Combined probability distribution
        """
        markov_probs = self.markov.predict_proba(context)
        rhythm_probs = self.rhythm.predict_proba(position)
        
        # Combine probabilities
        all_tokens = set(markov_probs.keys()) | set(rhythm_probs.keys())
        combined = {}
        
        for token in all_tokens:
            mp = markov_probs.get(token, 0)
            rp = rhythm_probs.get(token, 0)
            combined[token] = self.markov_weight * mp + (1 - self.markov_weight) * rp
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        
        return combined
    
    def sample(
        self,
        context: List[int],
        position: int,
        temperature: float = 1.0
    ) -> int:
        """Sample from combined model.
        
        Args:
            context: Previous tokens
            position: Current position in bar
            temperature: Sampling temperature
            
        Returns:
            Sampled token ID
        """
        probs = self.predict_proba(context, position)
        
        if not probs:
            return 0
        
        tokens = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # Apply temperature
        if temperature != 1.0:
            probabilities = probabilities ** (1.0 / temperature)
            probabilities /= probabilities.sum()
        
        return np.random.choice(tokens, p=probabilities)
    
    def generate(
        self,
        seed: List[int],
        max_length: int = 256,
        temperature: float = 1.0,
        eos_token: int = 2,
    ) -> List[int]:
        """Generate sequence with combined model.
        
        Args:
            seed: Initial tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            eos_token: End-of-sequence token ID
            
        Returns:
            Generated sequence
        """
        sequence = list(seed)
        position = 0
        
        for _ in range(max_length):
            next_token = self.sample(sequence, position, temperature)
            sequence.append(next_token)
            position += 1
            
            if next_token == eos_token:
                break
        
        return sequence
