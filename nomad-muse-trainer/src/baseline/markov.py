"""Variable-order Markov model with Witten-Bell smoothing.

Implements a simple but effective baseline for symbolic music generation.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


class VariableOrderMarkov:
    """Variable-order Markov model with backoff and smoothing."""
    
    def __init__(self, max_order: int = 4, smoothing: str = "witten-bell"):
        """Initialize Markov model.
        
        Args:
            max_order: Maximum context length
            smoothing: Smoothing method ("witten-bell" or "laplace")
        """
        self.max_order = max_order
        self.smoothing = smoothing
        
        # Store n-gram counts for each order
        # ngrams[order][context] = {token: count}
        self.ngrams: Dict[int, Dict[Tuple, Dict[int, int]]] = {
            order: defaultdict(lambda: defaultdict(int))
            for order in range(max_order + 1)
        }
        
        # Total counts for each context
        self.context_totals: Dict[int, Dict[Tuple, int]] = {
            order: defaultdict(int)
            for order in range(max_order + 1)
        }
        
        # Vocabulary seen
        self.vocab = set()
    
    def train(self, sequences: List[List[int]]):
        """Train on token sequences.
        
        Args:
            sequences: List of token ID sequences
        """
        for seq in sequences:
            self.vocab.update(seq)
            
            # Build n-grams for all orders
            for order in range(self.max_order + 1):
                for i in range(order, len(seq)):
                    context = tuple(seq[i - order:i]) if order > 0 else ()
                    token = seq[i]
                    
                    self.ngrams[order][context][token] += 1
                    self.context_totals[order][context] += 1
    
    def _witten_bell_probability(
        self,
        token: int,
        context: Tuple,
        order: int
    ) -> float:
        """Compute Witten-Bell smoothed probability.
        
        Args:
            token: Token to predict
            context: Context tuple
            order: N-gram order
            
        Returns:
            Smoothed probability
        """
        if order < 0:
            # Base case: uniform distribution
            return 1.0 / max(len(self.vocab), 1)
        
        counts = self.ngrams[order].get(context, {})
        total = self.context_totals[order].get(context, 0)
        
        if total == 0:
            # Backoff to lower order
            new_context = context[1:] if len(context) > 0 else ()
            return self._witten_bell_probability(token, new_context, order - 1)
        
        # Witten-Bell smoothing
        token_count = counts.get(token, 0)
        num_types = len(counts)  # Number of unique tokens seen in this context
        
        # Probability mass reserved for unseen tokens
        lambda_val = num_types / (total + num_types)
        
        if token_count > 0:
            # Seen token
            prob_seen = (1 - lambda_val) * (token_count / total)
            return prob_seen
        else:
            # Unseen token: backoff
            new_context = context[1:] if len(context) > 0 else ()
            prob_backoff = lambda_val * self._witten_bell_probability(token, new_context, order - 1)
            return prob_backoff
    
    def predict_proba(self, context: List[int]) -> Dict[int, float]:
        """Predict probability distribution for next token.
        
        Args:
            context: Previous tokens
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        # Use up to max_order context
        context = context[-self.max_order:] if len(context) > self.max_order else context
        context_tuple = tuple(context)
        order = len(context)
        
        # Compute probabilities for all tokens in vocab
        probs = {}
        for token in self.vocab:
            probs[token] = self._witten_bell_probability(token, context_tuple, order)
        
        # Normalize (should already be normalized, but just in case)
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        
        return probs
    
    def sample(self, context: List[int], temperature: float = 1.0) -> int:
        """Sample next token given context.
        
        Args:
            context: Previous tokens
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Sampled token ID
        """
        probs = self.predict_proba(context)
        
        if not probs:
            return 0
        
        tokens = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # Apply temperature
        if temperature != 1.0:
            probabilities = probabilities ** (1.0 / temperature)
            probabilities /= probabilities.sum()
        
        # Sample
        return np.random.choice(tokens, p=probabilities)
    
    def generate(
        self,
        seed: List[int],
        max_length: int = 256,
        temperature: float = 1.0,
        eos_token: int = 2,
    ) -> List[int]:
        """Generate a sequence.
        
        Args:
            seed: Initial tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            eos_token: End-of-sequence token ID
            
        Returns:
            Generated sequence
        """
        sequence = list(seed)
        
        for _ in range(max_length):
            next_token = self.sample(sequence, temperature)
            sequence.append(next_token)
            
            if next_token == eos_token:
                break
        
        return sequence
