"""Quantization and ONNX export.

Converts trained PyTorch model to int8 quantized ONNX for CPU inference.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml

from .model import TinyGRU, MiniTransformer
from .vocab import MusicVocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply dynamic int8 quantization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Quantized model
    """
    logger.info("Applying dynamic int8 quantization...")
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU, nn.LSTM},
        dtype=torch.qint8
    )
    
    return quantized_model


def export_to_onnx(model, out_path, vocab_size, seq_len, opset=17):
    import torch
    import torch.nn as nn

    model.eval()

    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            # Force a clean tensor-only signature + tensor-only return
            out = self.m(input_ids)

            # Common patterns:
            # - out is logits tensor
            # - out is tuple (logits, state)
            # - out is dict {"logits": ..., ...}
            if isinstance(out, (tuple, list)):
                logits = out[0]
            elif isinstance(out, dict):
                logits = out.get("logits", None)
                if logits is None:
                    # Fall back to first value
                    logits = next(iter(out.values()))
            else:
                logits = out

            return logits

    wrapper = OnnxWrapper(model).cpu()

    # Dummy input (batch=1, seq_len)
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    input_names = ["input_ids"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        out_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    
    logger.info(f"ONNX model saved to {out_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")


def compute_model_stats(model: nn.Module, vocab_size: int) -> dict:
    """Compute model statistics.
    
    Args:
        model: PyTorch model
        vocab_size: Vocabulary size
        
    Returns:
        Statistics dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    stats = {
        'model_type': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'vocab_size': vocab_size,
    }
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantize and export model to ONNX")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Config file")
    parser.add_argument("--no_quantize", action='store_true', help="Skip quantization")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
    # Reconstruct model from checkpoint config
    ckpt_config = checkpoint.get('config', config)
    model_state = checkpoint['model_state_dict']
    
    # Infer model type from state dict
    if any('gru' in key for key in model_state.keys()):
        model_type = 'gru'
    else:
        model_type = 'transformer'
    
    logger.info(f"Detected model type: {model_type}")
    
    # Get vocab size from state dict
    vocab_size = model_state['embedding.weight'].shape[0]
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create model
    from .model import create_model
    model = create_model(model_type, vocab_size, ckpt_config)
    model.load_state_dict(model_state)
    model.eval()
    
    # Quantize
    if not args.no_quantize:
        model = quantize_model(model)
    
    # Export to ONNX
    export_config = config.get('export', {})
    seq_len = config.get('data', {}).get('sequence_length', 512)
    opset = export_config.get('onnx_opset', 17)
    
    export_to_onnx(model, args.out, vocab_size, seq_len, opset)
    
    # Compute and save statistics
    stats = compute_model_stats(model, vocab_size)
    
    # Load existing stats if available
    stats_path = args.out.parent / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            existing_stats = json.load(f)
        existing_stats['model_stats'] = stats
        stats = existing_stats
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Model statistics saved to {stats_path}")
    logger.info(f"Model size: {stats['model_size_mb']:.2f} MB")
    logger.info(f"Parameters: {stats['total_parameters']:,}")
    
    # Save vocabulary if not already saved
    vocab_path = args.out.parent / "vocab.json"
    if not vocab_path.exists():
        logger.info(f"Saving vocabulary to {vocab_path}")
        from .vocab import create_vocab_from_config
        vocab = create_vocab_from_config(config)
        vocab.save(vocab_path)


if __name__ == "__main__":
    main()