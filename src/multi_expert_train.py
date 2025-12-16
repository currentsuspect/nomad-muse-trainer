"""Multi-Expert Training Pipeline for Nomad Muse Trainer.

Implements training for the multi-expert system:
- Individual expert training (drums, harmony, melody)
- Coordinated multi-expert training
- Expert-specific evaluation
- Reproducible training with fingerprints
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from expert_models import MultiExpertModel, create_expert_model, count_expert_parameters
from tokenizers_drums import create_drums_tokenizer, MuseDrumsTokenizer
from tokenizers_harmony import create_harmony_tokenizer, MuseHarmonyTokenizer
from tokenizers_melody import create_melody_tokenizer, MuseMelodyTokenizer
from nomad_format import create_nomad_format, NomadFormatBuilder

logger = logging.getLogger(__name__)


class ExpertDataset(Dataset):
    """Dataset for training individual experts."""
    
    def __init__(self, nomad_data: List[Dict], tokenizer: Any, sequence_length: int = 512):
        """Initialize dataset.
        
        Args:
            nomad_data: List of Nomad Symbolic Format dictionaries
            tokenizer: Expert tokenizer
            sequence_length: Length of sequences for training
        """
        self.nomad_data = nomad_data
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Tokenize all pieces
        self.sequences = []
        for data in nomad_data:
            if hasattr(tokenizer, 'tokenize_drum_track'):
                tokens = tokenizer.tokenize_drum_track(data)
            elif hasattr(tokenizer, 'tokenize_harmony_track'):
                tokens = tokenizer.tokenize_harmony_track(data)
            elif hasattr(tokenizer, 'tokenize_melody_track'):
                tokens = tokenizer.tokenize_melody_track(data)
            else:
                continue
            
            if len(tokens) < 2:
                continue
            
            # Encode to IDs
            if hasattr(tokenizer, 'vocab'):
                token_ids = tokenizer.vocab.encode_sequence(tokens)
            else:
                # Fallback for tokenizers without vocab
                logger.warning(f"Tokenizer {type(tokenizer).__name__} has no vocab attribute")
                continue
            
            # Create chunks - handle sequences shorter than sequence_length
            if len(token_ids) >= sequence_length:
                # Normal case: create overlapping chunks
                for i in range(0, len(token_ids) - sequence_length + 1, sequence_length // 2):
                    chunk = token_ids[i:i + sequence_length]
                    if len(chunk) == sequence_length:
                        self.sequences.append(chunk)
            else:
                # Short sequence: pad to sequence_length
                if len(token_ids) >= 2:  # Need at least 2 tokens for training
                    # Pad with padding token if available, otherwise repeat last token
                    pad_id = getattr(tokenizer.vocab, 'pad_id', token_ids[-1])
                    padded_ids = token_ids + [pad_id] * (sequence_length - len(token_ids))
                    self.sequences.append(padded_ids[:sequence_length])
        
        logger.info(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training example.
        
        Returns:
            (input_sequence, target_sequence) where target is input shifted by 1
        """
        seq = self.sequences[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_seq, target_seq


class MultiExpertTrainer:
    """Trainer for the multi-expert system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Auto-detect device - try CUDA first, fallback to CPU
        requested_device = config.get("training", {}).get("device", "cpu")
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU (CUDA) for training")
        elif requested_device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for training")
        
        # Initialize tokenizers
        self.tokenizers = self._init_tokenizers()
        
        # Initialize models
        self.experts = self._init_experts()
        
        # Training state
        self.training_config = config.get("training", {})
        self.best_losses = {}
        
        logger.info(f"Multi-expert trainer initialized on {self.device}")
    
    def _init_tokenizers(self) -> Dict[str, Any]:
        """Initialize expert tokenizers."""
        tokenizers = {}
        
        try:
            tokenizers["drums"] = create_drums_tokenizer(self.config)
            logger.info("Drums tokenizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize drums tokenizer: {e}")
        
        try:
            tokenizers["harmony"] = create_harmony_tokenizer(self.config)
            logger.info("Harmony tokenizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize harmony tokenizer: {e}")
        
        try:
            tokenizers["melody"] = create_melody_tokenizer(self.config)
            logger.info("Melody tokenizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize melody tokenizer: {e}")
        
        return tokenizers
    
    def _init_experts(self) -> MultiExpertModel:
        """Initialize expert models."""
        vocabularies = {}
        
        # Get vocabulary sizes
        for name, tokenizer in self.tokenizers.items():
            if hasattr(tokenizer, 'vocab'):
                vocabularies[name] = tokenizer.vocab.vocab_size
        
        # Create multi-expert system
        multi_expert = MultiExpertModel(self.config)
        multi_expert.build_experts(vocabularies)
        
        # Move to device
        if multi_expert.drums_model:
            multi_expert.drums_model.to(self.device)
        if multi_expert.harmony_model:
            multi_expert.harmony_model.to(self.device)
        if multi_expert.melody_model:
            multi_expert.melody_model.to(self.device)
        
        # Log parameter counts
        total_params = 0
        for name, model in [("drums", multi_expert.drums_model), 
                           ("harmony", multi_expert.harmony_model),
                           ("melody", multi_expert.melody_model)]:
            if model:
                params = count_expert_parameters(model)
                total_params += params
                logger.info(f"{name.capitalize()} model: {params:,} parameters")
        
        logger.info(f"Total parameters: {total_params:,}")
        return multi_expert
    
    def train_expert(self, expert_type: str, nomad_data: List[Dict], 
                    epochs: Optional[int] = None) -> Dict[str, float]:
        """Train a single expert model.
        
        Args:
            expert_type: "drums", "harmony", or "melody"
            nomad_data: Training data in Nomad format
            epochs: Number of epochs (uses config default if None)
            
        Returns:
            Training statistics
        """
        if expert_type not in self.tokenizers:
            raise ValueError(f"No tokenizer available for expert type: {expert_type}")
        
        tokenizer = self.tokenizers[expert_type]
        model = getattr(self.experts, f"{expert_type}_model")
        
        if model is None:
            raise ValueError(f"No model available for expert type: {expert_type}")
        
        # Training configuration
        epochs = epochs or self.training_config.get("epochs", 20)
        batch_size = self.training_config.get("batch_size", 16)
        learning_rate = self.training_config.get("learning_rate", 0.001)
        patience = self.training_config.get("early_stop_patience", 5)
        
        # Create dataset and dataloader
        sequence_length = self.config.get("data", {}).get("sequence_length", 64)  # Shorter for testing
        dataset = ExpertDataset(nomad_data, tokenizer, sequence_length)
        
        if len(dataset) == 0:
            raise ValueError(f"No valid training sequences for {expert_type} expert")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=self.training_config.get("num_workers", 0)
        )
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.pad_id)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        no_improvement = 0
        
        training_stats = {
            "expert_type": expert_type,
            "total_sequences": len(dataset),
            "vocab_size": tokenizer.vocab.vocab_size,
            "training_start": time.time()
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Training {expert_type} (epoch {epoch+1}/{epochs})")
            
            for batch_idx, (input_seq, target_seq) in enumerate(pbar):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits, hidden = model(input_seq)
                
                # Compute loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.training_config.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 self.training_config["grad_clip"])
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_epoch_loss = epoch_loss / num_batches
            training_stats[f"epoch_{epoch+1}_loss"] = avg_epoch_loss
            
            logger.info(f"{expert_type.capitalize()} Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                no_improvement = 0
                # Save best model
                self._save_expert_checkpoint(expert_type, model, epoch+1, avg_epoch_loss)
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Early stopping for {expert_type} after {epoch+1} epochs")
                    break
        
        training_stats["final_loss"] = best_loss
        training_stats["total_epochs"] = epoch + 1
        training_stats["training_time"] = time.time() - training_stats["training_start"]
        
        return training_stats
    
    def train_all_experts(self, nomad_data: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Train all expert models.
        
        Args:
            nomad_data: Training data in Nomad format
            
        Returns:
            Training statistics for all experts
        """
        all_stats = {}
        
        for expert_type in ["drums", "harmony", "melody"]:
            if expert_type in self.tokenizers:
                try:
                    logger.info(f"Starting training for {expert_type} expert")
                    stats = self.train_expert(expert_type, nomad_data)
                    all_stats[expert_type] = stats
                    logger.info(f"Completed training for {expert_type} expert")
                except Exception as e:
                    logger.error(f"Failed to train {expert_type} expert: {e}")
                    continue
        
        return all_stats
    
    def evaluate_expert(self, expert_type: str, nomad_data: List[Dict]) -> Dict[str, float]:
        """Evaluate a trained expert model.
        
        Args:
            expert_type: "drums", "harmony", or "melody"
            nomad_data: Evaluation data in Nomad format
            
        Returns:
            Evaluation metrics
        """
        tokenizer = self.tokenizers.get(expert_type)
        model = getattr(self.experts, f"{expert_type}_model")
        
        if tokenizer is None or model is None:
            raise ValueError(f"Tokenizer or model not available for {expert_type}")
        
        # Create evaluation dataset
        sequence_length = self.config.get("data", {}).get("sequence_length", 512)
        dataset = ExpertDataset(nomad_data, tokenizer, sequence_length)
        
        if len(dataset) == 0:
            return {"error": "No valid evaluation sequences"}
        
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Evaluation
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.pad_id, reduction='sum')
        
        with torch.no_grad():
            for input_seq, target_seq in tqdm(dataloader, desc=f"Evaluating {expert_type}"):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Forward pass
                logits, _ = model(input_seq)
                
                # Compute metrics
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
                total_loss += loss.item()
                
                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = target_seq != tokenizer.vocab.pad_id
                correct = (predictions == target_seq) & mask
                correct_predictions += correct.sum().item()
                total_tokens += mask.sum().item()
        
        # Compute final metrics
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        
        eval_metrics = {
            "expert_type": expert_type,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "total_loss": total_loss,
            "total_tokens": total_tokens
        }
        
        return eval_metrics
    
    def _save_expert_checkpoint(self, expert_type: str, model: nn.Module, 
                               epoch: int, loss: float):
        """Save checkpoint for an expert model."""
        checkpoint_dir = Path(self.training_config.get("checkpoint_dir", "artifacts/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "expert_type": expert_type,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": loss,
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / f"{expert_type}_best.pt")
    
    def generate_expert_fingerprint(self, expert_type: str) -> str:
        """Generate a reproducible fingerprint for an expert.
        
        Args:
            expert_type: Type of expert
            
        Returns:
            SHA256 hash string
        """
        # Create fingerprint data
        fingerprint_data = {
            "expert_type": expert_type,
            "config": self.config,
            "vocab_size": self.tokenizers[expert_type].vocab.vocab_size if expert_type in self.tokenizers else 0,
            "model_params": count_expert_parameters(getattr(self.experts, f"{expert_type}_model")) if getattr(self.experts, f"{expert_type}_model") else 0
        }
        
        # Create hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def prepare_nomad_dataset(midi_dir: Path, output_dir: Path, config: dict) -> List[Dict]:
    """Prepare dataset in Nomad Symbolic Format.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Directory to save Nomad format files
        config: Configuration dictionary
        
    Returns:
        List of Nomad format dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find MIDI files
    midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files")
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")
    
    # Convert to Nomad format
    nomad_data = []
    builder = NomadFormatBuilder()
    
    for midi_path in tqdm(midi_files, desc="Converting to Nomad format"):
        try:
            nomad_format = builder.from_midi(midi_path)
            
            # Save individual file
            file_output = output_dir / f"{midi_path.stem}_nomad.json"
            builder.save(nomad_format, file_output)
            
            nomad_data.append(nomad_format)
            
        except Exception as e:
            logger.warning(f"Failed to convert {midi_path}: {e}")
            continue
    
    logger.info(f"Successfully converted {len(nomad_data)} files to Nomad format")
    return nomad_data


def main():
    """CLI entry point for multi-expert training."""
    parser = argparse.ArgumentParser(description="Train multi-expert music models")
    parser.add_argument("--midi_dir", type=Path, required=True, help="Directory with MIDI files")
    parser.add_argument("--out", type=Path, default="artifacts", help="Output directory")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--expert", choices=["drums", "harmony", "melody", "all"], 
                       default="all", help="Which expert to train")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--eval_only", action="store_true", help="Evaluation only")
    parser.add_argument("--prepare_only", action="store_true", help="Dataset preparation only")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Prepare dataset
    if args.prepare_only or not args.eval_only:
        logger.info("Preparing Nomad dataset...")
        nomad_data = prepare_nomad_dataset(args.midi_dir, args.out / "nomad_data", config)
        
        if args.prepare_only:
            logger.info("Dataset preparation complete")
            return
    
    # Load prepared data
    nomad_data_path = args.out / "nomad_data"
    if nomad_data_path.exists():
        nomad_data = []
        for file_path in nomad_data_path.glob("*_nomad.json"):
            try:
                with open(file_path) as f:
                    nomad_data.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
    else:
        logger.error(f"Nomad data directory not found: {nomad_data_path}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = MultiExpertTrainer(config)
    
    # Training or evaluation
    if not args.eval_only:
        if args.expert == "all":
            logger.info("Training all experts...")
            stats = trainer.train_all_experts(nomad_data)
        else:
            logger.info(f"Training {args.expert} expert...")
            stats = trainer.train_expert(args.expert, nomad_data, args.epochs)
        
        # Save training stats
        stats_path = args.out / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training complete. Stats saved to {stats_path}")
    else:
        # Evaluation only
        if args.expert == "all":
            for expert_type in ["drums", "harmony", "melody"]:
                if expert_type in trainer.tokenizers:
                    logger.info(f"Evaluating {expert_type} expert...")
                    eval_metrics = trainer.evaluate_expert(expert_type, nomad_data)
                    logger.info(f"{expert_type.capitalize()} metrics: {eval_metrics}")
        else:
            logger.info(f"Evaluating {args.expert} expert...")
            eval_metrics = trainer.evaluate_expert(args.expert, nomad_data)
            logger.info(f"{args.expert.capitalize()} metrics: {eval_metrics}")


if __name__ == "__main__":
    main()