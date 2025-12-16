"""Run management for Muse Student Report Card system.

Handles run fingerprinting, logging, and artifact management.
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import yaml


def generate_run_id(model_type: str, dataset_path: Path, checkpoint_path: Optional[Path] = None) -> str:
    """Generate unique run ID.
    
    Format: YYYY-MM-DD_HHMMSS_<model>_<short_hash>
    
    Args:
        model_type: Type of model (gru, transformer)
        dataset_path: Path to dataset
        checkpoint_path: Optional path to checkpoint for git hash
        
    Returns:
        Unique run identifier
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    
    # Get short hash from git commit or dataset hash
    short_hash = get_short_hash(dataset_path, checkpoint_path)
    
    run_id = f"{timestamp}_{model_type}_{short_hash}"
    return run_id


def get_short_hash(dataset_path: Path, checkpoint_path: Optional[Path] = None) -> str:
    """Get short hash for run ID.
    
    Uses git commit hash if available, otherwise dataset hash.
    
    Args:
        dataset_path: Path to dataset
        checkpoint_path: Optional checkpoint path to check for git repo
        
    Returns:
        8-character hash string
    """
    # Try git commit hash first
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=checkpoint_path.parent if checkpoint_path else Path.cwd(),
            capture_output=True, 
            text=True, 
            check=True
        )
        git_hash = result.stdout.strip()
        return git_hash[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fall back to dataset hash
    try:
        dataset_hash = compute_dataset_fingerprint(dataset_path)["dataset_sha256"]
        return dataset_hash[:8]
    except Exception:
        # Last resort - use timestamp
        return datetime.now().strftime("%H%M%S")


def compute_dataset_fingerprint(dataset_path: Path) -> Dict[str, Any]:
    """Compute fingerprint for dataset.
    
    Args:
        dataset_path: Path to dataset.npz
        
    Returns:
        Dictionary with dataset fingerprint information
    """
    fingerprint = {}
    
    # Main dataset file
    if dataset_path.exists():
        with open(dataset_path, 'rb') as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()
        fingerprint["dataset_sha256"] = dataset_hash
        fingerprint["dataset_size"] = dataset_path.stat().st_size
    
    # Check for associated files
    vocab_path = dataset_path.parent / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'rb') as f:
            vocab_hash = hashlib.sha256(f.read()).hexdigest()
        fingerprint["vocab_sha256"] = vocab_hash
    
    stats_path = dataset_path.parent / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'rb') as f:
            stats_hash = hashlib.sha256(f.read()).hexdigest()
        fingerprint["stats_sha256"] = stats_hash
    
    return fingerprint


def create_run_directory(run_id: str, base_dir: Path = Path("artifacts/runs")) -> Path:
    """Create run directory structure.
    
    Args:
        run_id: Unique run identifier
        base_dir: Base directory for runs
        
    Returns:
        Path to run directory
    """
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    
    return run_dir


def save_config_resolved(config: Dict[str, Any], run_dir: Path):
    """Save resolved configuration.
    
    Args:
        config: Resolved configuration dictionary
        run_dir: Run directory path
    """
    config_path = run_dir / "config_resolved.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def save_environment_info(run_dir: Path):
    """Save environment information.
    
    Args:
        run_dir: Run directory path
    """
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "num_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now().isoformat(),
    }
    
    env_path = run_dir / "env.json"
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)


def save_dataset_fingerprint(dataset_path: Path, run_dir: Path):
    """Save dataset fingerprint.
    
    Args:
        dataset_path: Path to dataset
        run_dir: Run directory path
    """
    fingerprint = compute_dataset_fingerprint(dataset_path)
    fingerprint["dataset_path"] = str(dataset_path)
    
    fingerprint_path = run_dir / "dataset_fingerprint.json"
    with open(fingerprint_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)


def save_metrics_epoch(metrics: Dict[str, Any], epoch: int, run_dir: Path):
    """Save metrics for a single epoch.
    
    Args:
        metrics: Dictionary with epoch metrics
        epoch: Epoch number
        run_dir: Run directory path
    """
    metrics_file = run_dir / "metrics.jsonl"
    
    # Add epoch to metrics
    metrics["epoch"] = epoch
    metrics["timestamp"] = datetime.now().isoformat()
    
    # Append to JSONL file
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def save_run_summary(summary: Dict[str, Any], run_dir: Path):
    """Save run summary.
    
    Args:
        summary: Summary dictionary
        run_dir: Run directory path
    """
    summary_path = run_dir / "summary.json"
    
    # Add metadata
    summary["run_timestamp"] = datetime.now().isoformat()
    summary["python_version"] = sys.version
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


class MetricsLogger:
    """Logger for training metrics with automatic saving."""
    
    def __init__(self, run_dir: Path):
        """Initialize metrics logger.
        
        Args:
            run_dir: Run directory path
        """
        self.run_dir = run_dir
        self.epoch_metrics = []
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  perplexity: float, learning_rate: float, epoch_time: float):
        """Log epoch metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            perplexity: Validation perplexity
            learning_rate: Learning rate
            epoch_time: Time for epoch in seconds
        """
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
        }
        
        self.epoch_metrics.append(metrics)
        save_metrics_epoch(metrics, epoch, self.run_dir)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get run summary from logged metrics.
        
        Returns:
            Summary dictionary
        """
        if not self.epoch_metrics:
            return {}
        
        val_losses = [m["val_loss"] for m in self.epoch_metrics]
        train_losses = [m["train_loss"] for m in self.epoch_metrics]
        epoch_times = [m["epoch_time"] for m in self.epoch_metrics]
        
        # Find best validation loss
        best_epoch_idx = np.argmin(val_losses)
        best_val_loss = val_losses[best_epoch_idx]
        best_epoch = self.epoch_metrics[best_epoch_idx]["epoch"]
        
        # Calculate total time
        total_time = sum(epoch_times)
        
        summary = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_perplexity": np.exp(best_val_loss),
            "total_time_seconds": total_time,
            "total_epochs": len(self.epoch_metrics),
            "avg_epoch_time": np.mean(epoch_times),
        }
        
        return summary