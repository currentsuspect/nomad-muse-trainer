# Example: Generate multiple samples with different temperatures

import argparse
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_samples(
    onnx_path: Path,
    output_dir: Path,
    temperatures: list = [0.5, 0.8, 1.0, 1.2, 1.5],
    length: int = 256,
    num_samples_per_temp: int = 3,
):
    """Generate multiple samples with different temperatures.
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save samples
        temperatures: List of temperatures to try
        length: Token length for each sample
        num_samples_per_temp: Number of samples per temperature
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for temp in temperatures:
        logger.info(f"Generating samples with temperature={temp}")
        
        for i in range(num_samples_per_temp):
            output_file = output_dir / f"sample_temp{temp}_n{i+1}.mid"
            
            cmd = [
                "python", "-m", "scripts.demo_sample",
                "--onnx", str(onnx_path),
                "--out", str(output_file),
                "--length", str(length),
                "--temperature", str(temp),
                "--seed", str(42 + i),  # Different seed for each sample
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"  Generated: {output_file.name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"  Failed: {e}")
    
    logger.info(f"\nAll samples saved to {output_dir}")
    logger.info(f"Total: {len(temperatures) * num_samples_per_temp} MIDI files")


def main():
    parser = argparse.ArgumentParser(description="Generate multiple samples")
    parser.add_argument("--onnx", type=Path, default="artifacts/muse_quantized.onnx")
    parser.add_argument("--out_dir", type=Path, default="artifacts/samples")
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()
    
    generate_samples(args.onnx, args.out_dir, length=args.length)


if __name__ == "__main__":
    main()
