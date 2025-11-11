#!/bin/bash
# Quick start example: Run the full pipeline on sample data

set -e  # Exit on error

echo "================================================"
echo "nomad-muse-trainer Quick Start"
echo "================================================"
echo ""

# Check if data directory exists
if [ ! -d "data" ] || [ -z "$(ls -A data/*.mid 2>/dev/null)" ]; then
    echo "❌ No MIDI files found in ./data directory"
    echo ""
    echo "Please add some MIDI files to ./data/ first."
    echo "You can download sample datasets from:"
    echo "  - Lakh MIDI Dataset: http://colinraffel.com/projects/lmd/"
    echo "  - FreeMIDI: https://freemidi.org/"
    echo ""
    exit 1
fi

# Count MIDI files
NUM_MIDI=$(find data -name "*.mid" -o -name "*.midi" | wc -l)
echo "✓ Found $NUM_MIDI MIDI files in ./data"
echo ""

# Step 1: Prepare dataset
echo "Step 1: Preparing dataset..."
python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz
echo "✓ Dataset prepared"
echo ""

# Step 2: Train model (GRU by default, only 5 epochs for quick demo)
echo "Step 2: Training TinyGRU model (5 epochs for demo)..."
python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 5
echo "✓ Model trained"
echo ""

# Step 3: Export to ONNX
echo "Step 3: Quantizing and exporting to ONNX..."
python -m src.quantize_export \
    --ckpt ./artifacts/checkpoints/best.pt \
    --out ./artifacts/muse_quantized.onnx
echo "✓ Model exported"
echo ""

# Step 4: Build baseline
echo "Step 4: Building baseline model..."
python -m src.baseline.export \
    --dataset ./artifacts/dataset.npz \
    --out ./artifacts/baseline.bin
echo "✓ Baseline built"
echo ""

# Step 5: Evaluate
echo "Step 5: Evaluating model..."
python -m src.evaluate \
    --dataset ./artifacts/dataset.npz \
    --onnx ./artifacts/muse_quantized.onnx \
    --sample
echo "✓ Evaluation complete"
echo ""

# Step 6: Generate demo sample
echo "Step 6: Generating demo sample..."
python -m scripts.demo_sample \
    --onnx ./artifacts/muse_quantized.onnx \
    --out ./artifacts/demo_sample.mid \
    --length 256 \
    --temperature 1.0
echo "✓ Sample generated"
echo ""

echo "================================================"
echo "✅ Quick start complete!"
echo "================================================"
echo ""
echo "Generated files:"
echo "  - artifacts/muse_quantized.onnx  (Quantized model for C++ inference)"
echo "  - artifacts/vocab.json           (Vocabulary mapping)"
echo "  - artifacts/baseline.bin         (Baseline Markov model)"
echo "  - artifacts/demo_sample.mid      (Generated MIDI sample)"
echo "  - artifacts/sample_eval.mid      (Evaluation sample)"
echo ""
echo "Next steps:"
echo "  1. Listen to the generated MIDI files"
echo "  2. Adjust config.yaml and retrain"
echo "  3. Try different model architectures (--model transformer)"
echo "  4. Generate more samples with different temperatures"
echo "  5. Integrate the ONNX model into your C++ DAW"
echo ""
echo "For more info, see README.md"
