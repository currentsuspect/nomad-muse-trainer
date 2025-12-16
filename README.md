# Nomad Muse Trainer

🎵 **Train tiny CPU-friendly music models from MIDI for real-time inference in C++ DAWs**

**by Dylan Makori**

A production-ready pipeline for training lightweight neural music models and exporting them as quantized ONNX files for 100% local, CPU-based inference. Designed for embedding in Digital Audio Workstations (DAWs) and other real-time music applications.

> **⚖️ License**: The source code is licensed under the MIT License. See [License](#license) section below.

## Features

- 🚀 **CPU-optimized models**: TinyGRU and MiniTransformer architectures
- 🔢 **INT8 quantization**: Dynamic quantization for minimal memory footprint
- 📦 **ONNX export**: Standard format for C++ integration
- 🎹 **MIDI tokenization**: Discrete vocab for symbolic music (notes, timing, velocity, duration)
- 📊 **Baseline models**: Markov chains with Witten-Bell smoothing for comparison
- 🎯 **Complete pipeline**: Data prep → Train → Quantize → Export → Evaluate
- 🔬 **Reproducible**: Config-based workflow with seed control

## Architecture

### Neural Models
- **TinyGRU**: 2-layer GRU (default: 128 hidden units, ~200K parameters)
- **MiniTransformer**: 2-layer transformer (2 heads, d_model=256, ~400K parameters)

### Tokenization
- **NOTE_ON_{0-127}**: Note start events
- **NOTE_OFF_{0-127}**: Note end events  
- **TIME_SHIFT_{1-32}**: Discrete time steps (default: 20ms bins)
- **VELOCITY_{1-8}**: Quantized MIDI velocity
- **DURATION_{1-16}**: Quantized note duration
- Special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`

### Baseline
- Variable-order Markov model (order 3-5) with Witten-Bell smoothing
- Bar-position rhythm histograms
- Combined model serialized to `.bin` for lightweight inference

## Installation

```bash
# Clone the repository
cd nomad-muse-trainer

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Requirements
- Python 3.11+
- PyTorch 2.0+
- ONNX Runtime
- pretty_midi / miditoolkit
- See `requirements.txt` for full list

## Quick Start

### 1. Prepare Your Data

Place MIDI files in a `data/` directory:

```bash
mkdir -p data
# Copy your MIDI files here
# Can use subdirectories for organization
```

### 2. Index MIDI Files (Optional)

Create a manifest with metadata and train/val/test splits:

```bash
python -m scripts.index_midi --midi_dir ./data --out ./artifacts/manifest.csv
```

### 3. Prepare Dataset

Convert MIDI files to tokenized sequences:

```bash
python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz
```

This will:
- Tokenize all valid MIDI files
- Create train/val/test splits (stratified by tempo/key)
- Save vocabulary to `artifacts/vocab.json`
- Save statistics to `artifacts/stats.json`

### 4. Train Model

**GRU model (faster, smaller):**
```bash
python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20
```

**Transformer model (better quality):**
```bash
python -m src.train --dataset ./artifacts/dataset.npz --model transformer --epochs 20
```

Training checkpoints saved to `artifacts/checkpoints/`:
- `best.pt`: Best model by validation loss
- `latest.pt`: Most recent checkpoint

### 5. Quantize and Export to ONNX

```bash
python -m src.quantize_export \
  --ckpt ./artifacts/checkpoints/best.pt \
  --out ./artifacts/muse_quantized.onnx
```

This produces:
- `artifacts/muse_quantized.onnx`: INT8 quantized model
- `artifacts/vocab.json`: Vocabulary mapping
- `artifacts/stats.json`: Model statistics

### 6. Build Baseline Model

```bash
python -m src.baseline.export \
  --dataset ./artifacts/dataset.npz \
  --out ./artifacts/baseline.bin
```

### 7. Evaluate

```bash
python -m src.evaluate \
  --dataset ./artifacts/dataset.npz \
  --onnx ./artifacts/muse_quantized.onnx \
  --sample
```

Outputs:
- Perplexity on test set
- Top-k accuracy (k=1,3,5,10)
- Sample MIDI generation

### 8. Generate Samples

```bash
python -m scripts.demo_sample \
  --onnx ./artifacts/muse_quantized.onnx \
  --out ./artifacts/sample.mid \
  --length 256 \
  --temperature 1.0
```

Parameters:
- `--length`: Number of tokens to generate
- `--temperature`: Sampling temperature (higher = more random)
- `--tempo`: Output MIDI tempo (BPM)

## Makefile Shortcuts

```bash
make help        # Show all commands
make prep        # Prepare dataset
make train       # Train GRU model
make train-tx    # Train transformer
make export      # Quantize and export to ONNX
make baseline    # Build baseline model
make eval        # Evaluate model
make demo        # Generate sample
make all         # Run full pipeline
make clean       # Remove artifacts
```

## Configuration

Edit `config.yaml` to customize:

### Tokenization
```yaml
tokenization:
  time_shift_bins: 32      # Time discretization
  time_shift_ms: 20        # Milliseconds per bin
  velocity_bins: 8         # Velocity quantization levels
  duration_bins: 16        # Duration quantization levels
  max_duration_ms: 2000    # Max note duration
```

### Data Preparation
```yaml
data:
  sequence_length: 512     # Training sequence length
  min_midi_length: 32      # Min tokens to include file
  max_midi_length: 16384   # Max tokens (truncate longer)
  melody_only: false       # Use only first track
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 42
```

### Training
```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  early_stop_patience: 5
  device: "cuda"           # or "cpu"
```

### Model Architectures
```yaml
models:
  gru:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
  
  transformer:
    d_model: 256
    nhead: 2
    num_layers: 2
    dim_feedforward: 512
    dropout: 0.2
```

## CLI Reference

### Data Preparation
```bash
python -m src.data_prep \
  --midi_dir <path>        # Directory with MIDI files
  --out <path>             # Output .npz file
  --config config.yaml     # Configuration file
```

### Training
```bash
python -m src.train \
  --dataset <path>         # Path to dataset.npz
  --model {gru,transformer}
  --epochs <int>           # Number of epochs (overrides config)
  --batch_size <int>       # Batch size (overrides config)
  --hidden <int>           # Hidden size for GRU
  --layers <int>           # Number of layers
  --resume <path>          # Resume from checkpoint
  --config config.yaml
```

### Quantization & Export
```bash
python -m src.quantize_export \
  --ckpt <path>            # Input checkpoint (.pt)
  --out <path>             # Output ONNX file
  --no_quantize            # Skip quantization (for debugging)
  --config config.yaml
```

### Evaluation
```bash
python -m src.evaluate \
  --dataset <path>         # Path to dataset.npz
  --onnx <path>            # Path to ONNX model
  --sample                 # Generate sample MIDI
  --config config.yaml
```

### Sample Generation
```bash
python -m scripts.demo_sample \
  --onnx <path>            # Path to ONNX model
  --out <path>             # Output MIDI file
  --length <int>           # Tokens to generate (default: 256)
  --temperature <float>    # Sampling temperature (default: 1.0)
  --tempo <int>            # MIDI tempo in BPM (default: 120)
  --seed <int>             # Random seed (default: 42)
```

## Output Files

After running the full pipeline, you'll have:

```
artifacts/
├── dataset.npz              # Tokenized training data
├── vocab.json               # Vocabulary mapping
├── stats.json               # Dataset & model statistics
├── muse_quantized.onnx      # INT8 quantized model
├── baseline.bin             # Baseline Markov model
├── checkpoints/
│   ├── best.pt              # Best PyTorch checkpoint
│   └── latest.pt            # Latest PyTorch checkpoint
├── sample.mid               # Generated MIDI sample
└── sample_eval.mid          # Evaluation sample
```

## C++ Integration Example

```cpp
// Load ONNX model
Ort::Env env;
Ort::SessionOptions options;
Ort::Session session(env, "artifacts/muse_quantized.onnx", options);

// Prepare input
std::vector<int64_t> input_ids = {1, 132, 64, ...};  // Token IDs
std::vector<int64_t> input_shape = {1, input_ids.size()};

// Run inference
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
    memory_info, input_ids.data(), input_ids.size(), 
    input_shape.data(), input_shape.size()
);

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names, &input_tensor, 1,
    output_names, 1
);

// Get logits and sample
float* logits = output_tensors[0].GetTensorMutableData<float>();
// Apply softmax and sample...
```

## Tips & Best Practices

### Data Quality
- Use clean, well-aligned MIDI files
- Mix different styles/tempos for better generalization
- Aim for 1000+ MIDI files for good coverage
- Remove very short or very long files (use `min_midi_length` and `max_midi_length`)

### Training
- Start with GRU for faster iteration
- Use early stopping to avoid overfitting
- Monitor validation loss closely
- Try different learning rates (0.0001 - 0.01)
- Increase `sequence_length` for longer-range dependencies

### Model Size
- GRU with hidden=128: ~1-2 MB quantized
- Transformer with d_model=256: ~2-4 MB quantized
- Adjust `hidden_size` or `d_model` to fit your constraints

### Sampling
- Temperature 0.8-1.2 works well for music
- Lower temperature (0.5-0.8): More predictable, safer
- Higher temperature (1.2-1.5): More creative, riskier
- Use primer sequences from your target style

## Troubleshooting

### "No MIDI files found"
- Check your `--midi_dir` path
- Ensure files have `.mid` or `.midi` extension
- Run `index_midi.py` to validate files

### "CUDA out of memory"
- Reduce `batch_size` in config
- Reduce `sequence_length`
- Use `--device cpu` in training config

### "Model perplexity is very high"
- Train for more epochs
- Check data quality
- Ensure vocabulary matches training data
- Try different learning rate

### "Generated MIDI sounds random"
- Lower sampling temperature
- Train longer
- Use more/better training data
- Try different primer sequences

## Roadmap

- [ ] Support for percussion/drum tracks
- [ ] Multi-track generation
- [ ] Static quantization support
- [ ] TorchScript export option
- [ ] Pre-trained model zoo
- [ ] VST3 plugin example

## License

**Nomad Muse Trainer** is licensed under the **MIT License**.

This means you are free to:

- ✓ Use the code for any purpose, including commercial
- ✓ Modify and extend the training pipeline
- ✓ Distribute the source code with attribution
- ✓ Train your own models on your own data
- ✓ Use the code in commercial projects

See the [LICENSE](LICENSE) file for complete terms.

### Quick Summary

The MIT License is a permissive license that allows you to use, modify, and distribute this software freely. The only requirements are:
- Include the original copyright notice
- Include the license text in copies or substantial portions

### What This Means

You can use this software to train your own music models, modify the code for your needs, and even build commercial products with it. The trained models and datasets you create are your own to use and distribute as you wish.

## Citation

If you use this code in your research or products, please cite:

```bibtex
@software{nomad_muse_trainer,
  title = {Nomad Muse Trainer: Lightweight Music Model Training Pipeline},
  author = {Dylan Makori},
  year = {2025},
  url = {https://github.com/nomadstudios/nomad-muse-trainer},
  note = {Code licensed under MIT}
}
```

## Contributing

Contributions to the open-source code are welcome! Please:
- Open an issue for bugs or feature requests
- Submit pull requests for code improvements
- Follow the existing code style
- Ensure changes don't include proprietary assets

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

- Built with PyTorch, ONNX Runtime, and pretty_midi
- Inspired by Music Transformer, MuseNet, and Magenta
- Thanks to the open-source music generation community

---

**Nomad Muse Trainer** is a product of **Dylan Makori**  
Copyright © 2025 Dylan Makori. All rights reserved.

---

**Made with ❤️ for real-time music generation**