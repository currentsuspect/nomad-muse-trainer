#!/bin/bash
# Multi-Expert Nomad Muse Trainer Setup Script

set -e

echo "🎵 Setting up Multi-Expert Nomad Muse Trainer..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc) -eq 0 ]]; then
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip

# Core dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pretty_midi pyyaml tqdm

# Optional dependencies for advanced features
pip install onnxruntime  # For ONNX export
pip install matplotlib seaborn  # For visualization

# Development dependencies
pip install pytest black flake8  # For testing and code quality

echo "✅ Dependencies installed"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p artifacts/checkpoints
mkdir -p artifacts/models
mkdir -p artifacts/logs

# Copy configuration files
if [ ! -f "config.yaml" ]; then
    cp config_multi_expert.yaml config.yaml
    echo "✅ Configuration file created"
fi

# Run tests to verify installation
echo "🧪 Running tests..."
python test_simple.py
python test_multi_expert.py

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your MIDI files to data/raw/"
echo "2. Run: python test_multi_expert.py"
echo "3. Train models: python src/multi_expert_train.py --midi_dir data/raw --expert all"
echo ""
echo "For more information, see README.md and MULTI_EXPERT_README.md"