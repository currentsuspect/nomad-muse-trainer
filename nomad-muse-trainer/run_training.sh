#!/bin/bash
# Training script for Nomad Muse Trainer

set -e  # Exit on error

echo "=================================="
echo "Nomad Muse Trainer - Training Run"
echo "=================================="
echo ""

# Change to project directory
cd "$(dirname "$0")"

# Use the venv python
PYTHON=/workspaces/codespaces-blank/.venv/bin/python

echo "Step 1: Preparing dataset..."
echo "This will tokenize all MIDI files (~10-15 minutes)"
echo ""

$PYTHON -m src.data_prep \
  --midi_dir ./data \
  --out ./artifacts/dataset.npz

echo ""
echo "✅ Dataset prepared!"
echo ""
echo "Step 2: Training GRU model..."
echo "Expected time: ~30-40 minutes for 20 epochs"
echo ""

$PYTHON -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model gru \
  --epochs 20

echo ""
echo "=================================="
echo "✅ Training Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Export to ONNX:"
echo "   $PYTHON -m src.quantize_export \\"
echo "     --checkpoint artifacts/checkpoints/gru/best.pt \\"
echo "     --vocab artifacts/vocab.json \\"
echo "     --out artifacts/nomad_muse_gru_int8.onnx"
echo ""
echo "2. Generate sample:"
echo "   $PYTHON scripts/demo_sample.py \\"
echo "     --model artifacts/nomad_muse_gru_int8.onnx \\"
echo "     --vocab artifacts/vocab.json \\"
echo "     --out sample.mid"
echo ""
