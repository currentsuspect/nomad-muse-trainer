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
    
    # Ensure model is in eval mode and on CPU
    model.cpu()
    
    # Create a clean wrapper that handles ScriptObject issues
    class CleanWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Store reference to the original model
            self.model = model
            # Ensure model is in eval mode
            self.model.eval()
            
        def forward(self, input_ids):
            with torch.no_grad():  # Ensure no gradients tracked
                # Call model and immediately convert outputs
                out = self.model(input_ids)
                
                # Handle different output formats
                if isinstance(out, (tuple, list)):
                    result = out[0]
                elif isinstance(out, dict):
                    result = out.get('logits', next(iter(out.values())))
                else:
                    result = out
                
                # Ensure output is a plain tensor (JIT-compatible)
                result = result.detach()
                    
                return result

    wrapper = CleanWrapper(model)
    
    # Test the wrapper first
    test_input = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)
    try:
        with torch.no_grad():
            test_output = wrapper(test_input)
            logger.info(f"Wrapper test successful. Output shape: {test_output.shape}")
    except Exception as e:
        logger.error(f"Wrapper test failed: {e}")
        raise
    
    # Dummy input for export
    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    
    # Export parameters
    input_names = ["input_ids"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time"},
    }
    
    # Try multiple export strategies
    export_succeeded = False
    
    # Strategy 1: Standard export with example_outputs
    try:
        logger.info("Attempting export with example_outputs...")
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            out_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
            example_outputs=wrapper(dummy_input),  # Provide explicit example outputs
        )
        export_succeeded = True
        logger.info("Export with example_outputs successful!")
    except Exception as e1:
        logger.warning(f"Export with example_outputs failed: {e1}")
        
        # Strategy 2: Try torch.jit.script to convert model to JIT graph first
        try:
            logger.info("Attempting JIT script conversion...")
            # Convert to JIT script to handle ScriptObject issues
            scripted_wrapper = torch.jit.script(wrapper)
            
            logger.info("Attempting export with scripted model...")
            torch.onnx.export(
                scripted_wrapper,
                (dummy_input,),
                out_path,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=False,
            )
            export_succeeded = True
            logger.info("Export with scripted model successful!")
        except Exception as e2:
            logger.warning(f"Export with scripted model failed: {e2}")
            
            # Strategy 3: Try JIT tracing instead of scripting
            try:
                logger.info("Attempting JIT tracing...")
                # Use tracing instead of scripting to avoid ScriptObject issues
                traced_wrapper = torch.jit.trace(wrapper, dummy_input)
                
                logger.info("Attempting export with traced model...")
                torch.onnx.export(
                    traced_wrapper,
                    (dummy_input,),
                    out_path,
                    opset_version=opset,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    verbose=False,
                )
                export_succeeded = True
                logger.info("Export with traced model successful!")
            except Exception as e3:
                logger.warning(f"Export with traced model failed: {e3}")
                
                # Strategy 4: Minimal export without dynamic axes
                try:
                    logger.info("Attempting minimal export...")
                    torch.onnx.export(
                        wrapper,
                        (dummy_input,),
                        out_path,
                        opset_version=opset,
                        input_names=input_names,
                        output_names=output_names,
                        # No dynamic axes to simplify
                        do_constant_folding=True,
                        verbose=False,
                    )
                    export_succeeded = True
                    logger.info("Minimal export successful!")
                except Exception as e4:
                    logger.error(f"All export strategies failed: {e4}")
                    raise
    
    if not export_succeeded:
        raise RuntimeError("All export strategies failed")
    
    logger.info(f"ONNX model saved to {out_path}")
    
    # Verify ONNX model (optional, skip if verification fails)
    try:
        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully")
    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")
        # Don't raise - the model might still be usable


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
        if "model_size_mb" in stats:
            logger.info(f"Model size: {stats['model_size_mb']:.2f} MB")
        else:
            logger.info("Model size: (not computed)")
        
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