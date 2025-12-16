"""Training script with CLI, checkpointing, and early stopping."""

import os
import argparse
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from .model import create_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CPU OPTIMIZATION: Configure PyTorch for optimal CPU performance
# ============================================================================
def setup_cpu_optimizations(num_cores: int = 4):
    """Configure PyTorch for efficient CPU-only training.
    
    On a 4-core/16GB Codespace, we want to:
    1. Use all available cores for computation
    2. Disable GPU-specific optimizations
    3. Enable CPU-specific optimizations
    
    Args:
        num_cores: Number of CPU cores to use (default: 4 for Codespaces)
    """
    # Use all CPU cores for intra-op parallelism (matrix ops, etc.)
    torch.set_num_threads(num_cores)
    
    # Disable cuDNN (GPU-specific optimization)
    torch.backends.cudnn.enabled = False
    
    # Enable CPU-specific optimizations
    torch.set_flush_denormal(True)  # Faster floating point on CPU
    
    # Disable gradient accumulation warnings
    torch.autograd.set_detect_anomaly(False)
    
    logger.info(f"✓ CPU optimizations enabled:")
    logger.info(f"  - Threads: {torch.get_num_threads()}")
    logger.info(f"  - Device: CPU only")
    logger.info(f"  - Memory: 16 GB available")
    
    return num_cores


class Trainer:
    """Trainer for music generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: Path,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        train_config = config.get('training', {})
        
        # Use CUDA if available and configured, otherwise default to CPU
        device_str = train_config.get('device', 'cpu')
        if device_str == 'cuda' and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, falling back to CPU.")
            device_str = 'cpu'
        
        self.device = torch.device(device_str)
        self.model.to(self.device)
        
        logger.info("=" * 70)
        logger.info("MODEL CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Parameters: {count_parameters(model):,}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info("=" * 70)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.get('learning_rate', 0.001),
            weight_decay=train_config.get('weight_decay', 0.0001),
        )
        
        sched_kwargs = dict(
            optimizer=self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**sched_kwargs, verbose=True)
        except TypeError:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(**sched_kwargs)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is PAD
        
        # Training parameters
        self.epochs = train_config.get('epochs', 50)
        self.grad_clip = train_config.get('grad_clip', 1.0)
        self.early_stop_patience = train_config.get('early_stop_patience', 5)
        
        # State tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.training_start_time = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Trainer initialized")
        logger.info(f"  Checkpoints: {self.checkpoint_dir}")
        logger.info("")
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Track epoch timing for ETA calculation
        epoch_start = time.time()
        
        # Create progress bar with detailed info
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            ncols=100,  # Fixed width for stable display
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, batch in enumerate(pbar):
            sequences = batch[0].to(self.device)
            
            # Input and target (shifted by 1)
            inputs = sequences[:, :-1]
            targets = sequences[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, nn.Module) and hasattr(self.model, 'gru'):
                # GRU model
                logits, _ = self.model(inputs)
            else:
                # Transformer model
                logits = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients on CPU)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with current loss
            current_avg_loss = total_loss / num_batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{current_avg_loss:.4f}'
            })
            

        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        logger.info(f"  Train loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Validating", ncols=100, leave=False):
                sequences = batch[0].to(self.device)
                
                inputs = sequences[:, :-1]
                targets = sequences[:, 1:]
                
                if isinstance(self.model, nn.Module) and hasattr(self.model, 'gru'):
                    logits, _ = self.model(inputs)
                else:
                    logits = self.model(inputs)
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"  Val loss:   {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, force_save: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            force_save: Force save even if not best (for epoch checkpoints)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Always save latest checkpoint (Codespace-safe)
        if force_save or True:  # Always save for safety
            latest_path = self.checkpoint_dir / 'latest.pt'
            torch.save(checkpoint, latest_path)
            logger.info(f"  💾 Checkpoint saved: {latest_path.name}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"  ⭐ Best model saved: {best_path.name} (val_loss={self.best_val_loss:.4f})")
        
        # Save epoch-specific checkpoint every 5 epochs (for long training)
        if self.current_epoch % 5 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
            logger.info(f"  📌 Milestone saved: {epoch_path.name}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop with CPU optimizations and progress tracking."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)
        logger.info(f"Total epochs: {self.epochs}")
        logger.info(f"Early stopping patience: {self.early_stop_patience}")
        logger.info("")
        
        self.training_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Print epoch header
            logger.info("-" * 70)
            logger.info(f"EPOCH {epoch + 1}/{self.epochs}")
            logger.info("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                logger.info(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                logger.info(f"  ✓ New best! Improved by {improvement:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Save checkpoint (ALWAYS save for Codespace safety)
            self.save_checkpoint(is_best=is_best, force_save=True)
            
            # Calculate and display timing info
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - self.training_start_time
            
            # Estimate time remaining
            epochs_done = epoch + 1 - self.current_epoch
            if epochs_done > 0:
                avg_epoch_time = elapsed_total / epochs_done
                epochs_remaining = self.epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * epochs_remaining
                eta = timedelta(seconds=int(eta_seconds))
                
                logger.info(f"  ⏱️  Epoch time: {epoch_time:.1f}s | "
                           f"Total: {elapsed_total/60:.1f}min | "
                           f"ETA: {eta}")
            
            logger.info("")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.early_stop_patience:
                logger.info("=" * 70)
                logger.info(f"⚠️  EARLY STOPPING")
                logger.info(f"No improvement for {self.early_stop_patience} epochs")
                logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                logger.info("=" * 70)
                break
        
        # Training complete
        total_time = time.time() - self.training_start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Final epoch: {self.current_epoch + 1}/{self.epochs}")
        logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        logger.info("=" * 70)
        logger.info("")


def main():
    """Main entry point with CPU optimization."""
    parser = argparse.ArgumentParser(description="Train music generation model on CPU")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.npz")
    parser.add_argument("--model", type=str, choices=['gru', 'transformer'], required=True)
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config, 8-16 recommended for CPU)")
    parser.add_argument("--hidden", type=int, help="Hidden size for GRU (overrides config)")
    parser.add_argument("--layers", type=int, help="Number of layers (overrides config)")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config to determine device
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Determine device
    device = config.get('training', {}).get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Apply CPU optimizations only if using CPU
    if device == 'cpu':
        setup_cpu_optimizations()
    
    # Display header
    logger.info("")
    logger.info("=" * 70)
    if device == 'cuda':
        logger.info("🚀 GPU-ACCELERATED TRAINING MODE")
    else:
        logger.info("🖥️  CPU-ONLY TRAINING MODE")
    logger.info("=" * 70)
    logger.info("")

    # Override config with CLI args
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.hidden:
        config.setdefault('models', {}).setdefault('gru', {})['hidden_size'] = args.hidden
    if args.layers:
        config.setdefault('models', {}).setdefault(args.model, {})['num_layers'] = args.layers
    
    # Log device and environment info
    logger.info("Environment:")
    if device == 'cuda':
        logger.info(f"  • Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  • CUDA version: {torch.version.cuda}")
    else:
        logger.info(f"  • PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"  • Available CPU cores: {os.cpu_count()}")

    # Adjust batch size and workers based on device
    train_config = config.get('training', {})
    if device == 'cpu':
        if 'batch_size' not in train_config or train_config['batch_size'] > 16:
            logger.warning(f"  ⚠️  Large batch size may be slow on CPU. Recommend 8-16.")
        if 'num_workers' not in train_config or train_config['num_workers'] > 2:
            logger.info(f"  • DataLoader workers: 2 (optimized for CPU)")
            train_config['num_workers'] = 2
    else: # GPU
        if 'batch_size' not in train_config:
            train_config['batch_size'] = 64 # Default for GPU
        if 'num_workers' not in train_config:
            train_config['num_workers'] = min(os.cpu_count(), 8)
    
    logger.info(f"  • Batch size: {train_config.get('batch_size')}")
    logger.info(f"  • DataLoader workers: {train_config.get('num_workers')}")
    logger.info("")

    logger.info("Training configuration:")
    logger.info(f"  • Model: {args.model}")
    logger.info(f"  • Epochs: {train_config.get('epochs', 20)}")
    logger.info(f"  • Learning rate: {train_config.get('learning_rate', 0.001)}")
    logger.info("")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    data = np.load(args.dataset)
    
    train_data = torch.from_numpy(data['train']).long()
    val_data = torch.from_numpy(data['val']).long()
    vocab_size = int(data['vocab_size'][0])
    
    logger.info(f"  Train sequences: {len(train_data):,}")
    logger.info(f"  Validation sequences: {len(val_data):,}")
    logger.info(f"  Vocabulary size: {vocab_size:,}")
    logger.info("")
    
    # Create data loaders
    pin_memory = device == 'cuda'
    logger.info(f"DataLoaders created with pin_memory={'True' if pin_memory else 'False'}")

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=train_config.get('batch_size'),
        shuffle=True,
        num_workers=train_config.get('num_workers'),
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=train_config.get('batch_size'),
        shuffle=False,
        num_workers=train_config.get('num_workers'),
        pin_memory=pin_memory,
    )
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = create_model(args.model, vocab_size, config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {num_params:,}")
    logger.info("")
    
    # Create trainer
    checkpoint_dir = Path(train_config.get('checkpoint_dir', 'artifacts/checkpoints'))
    trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir)
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()