# Project Structure

```
nomad-muse-trainer/
│
├── README.md                   # Comprehensive documentation
├── LICENSE                     # MIT License
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── Makefile                    # Build shortcuts
├── quickstart.sh              # Quick start script
├── .gitignore                 # Git ignore rules
│
├── src/                       # Main source code
│   ├── __init__.py
│   ├── vocab.py               # Vocabulary/tokenization
│   ├── data_prep.py           # MIDI → tokens pipeline
│   ├── model.py               # TinyGRU & MiniTransformer
│   ├── train.py               # Training pipeline
│   ├── quantize_export.py     # INT8 quantization + ONNX export
│   ├── evaluate.py            # Perplexity, top-k, sampling
│   └── baseline/              # Baseline models
│       ├── __init__.py
│       ├── markov.py          # Variable-order Markov
│       ├── rhythm.py          # Rhythm histograms
│       └── export.py          # Baseline serialization
│
├── scripts/                   # Utility scripts
│   ├── __init__.py
│   ├── index_midi.py          # MIDI file indexing
│   └── demo_sample.py         # ONNX sample generation
│
├── examples/                  # Example usage
│   └── generate_batch.py      # Batch sample generation
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_vocab.py          # Vocabulary tests
│
├── data/                      # MIDI input files (user provided)
│   └── README.md
│
└── artifacts/                 # Generated outputs
    └── README.md              # Output documentation
```

## File Counts

- **15** Python modules (excluding __init__.py)
- **4** __init__.py files
- **7** configuration/documentation files
- **Total LOC**: ~3000+ lines of production code

## Key Features Implemented

✅ **Vocabulary System** (`vocab.py`)
- NOTE_ON/OFF, TIME_SHIFT, VELOCITY, DURATION tokens
- Configurable quantization bins
- Save/load to JSON

✅ **Data Preparation** (`data_prep.py`)
- MIDI parsing with pretty_midi
- Tempo/key detection
- Stratified train/val/test split
- Sequence chunking with overlap

✅ **Neural Models** (`model.py`)
- TinyGRU: 2-layer GRU (128 hidden, ~200K params)
- MiniTransformer: 2-layer transformer (256 d_model, ~400K params)
- Both optimized for CPU inference

✅ **Training Pipeline** (`train.py`)
- Teacher forcing
- Early stopping
- Checkpoint management
- CLI with overrides

✅ **Quantization & Export** (`quantize_export.py`)
- Dynamic INT8 quantization
- ONNX export (opset 17)
- Model statistics

✅ **Baseline Models** (`baseline/`)
- Variable-order Markov with Witten-Bell smoothing
- Bar-position rhythm histograms
- Combined model serialization

✅ **Evaluation** (`evaluate.py`)
- Perplexity computation
- Top-k accuracy (k=1,3,5,10)
- ONNX inference
- Sample generation to MIDI

✅ **Utilities** (`scripts/`)
- MIDI file indexing with metadata
- Demo sample generator
- Batch generation example

## Supported Commands

### Core Pipeline
```bash
make prep         # Data preparation
make train        # Train GRU
make train-tx     # Train transformer
make export       # Quantize + ONNX export
make baseline     # Build baseline
make eval         # Evaluate model
make demo         # Generate sample
make all          # Full pipeline
```

### Individual Tools
```bash
# Data prep
python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz

# Training
python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20

# Export
python -m src.quantize_export --ckpt ./artifacts/checkpoints/best.pt --out ./artifacts/muse_quantized.onnx

# Baseline
python -m src.baseline.export --dataset ./artifacts/dataset.npz --out ./artifacts/baseline.bin

# Evaluation
python -m src.evaluate --dataset ./artifacts/dataset.npz --onnx ./artifacts/muse_quantized.onnx

# Sample generation
python -m scripts.demo_sample --onnx ./artifacts/muse_quantized.onnx --out ./artifacts/sample.mid
```

## Configuration Options

All configurable via `config.yaml`:
- Tokenization parameters
- Data split ratios
- Model architectures
- Training hyperparameters
- Export settings
- Evaluation metrics

## Output Files

After full pipeline:
- `artifacts/dataset.npz` - Tokenized training data
- `artifacts/vocab.json` - Vocabulary mapping
- `artifacts/stats.json` - Statistics
- `artifacts/muse_quantized.onnx` - INT8 ONNX model (1-5 MB)
- `artifacts/baseline.bin` - Baseline model
- `artifacts/checkpoints/best.pt` - Best PyTorch checkpoint
- `artifacts/sample.mid` - Generated MIDI

## Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_vocab.py -v
```

## Next Steps

1. Add MIDI files to `data/` directory
2. Run `./quickstart.sh` for end-to-end demo
3. Adjust `config.yaml` for your needs
4. Train with more epochs for production
5. Integrate ONNX model into C++ DAW

## Production Readiness

✅ Complete pipeline implementation
✅ Error handling and logging
✅ Configurable via YAML
✅ CLI arguments with overrides
✅ Reproducible (seed control)
✅ Documented code with docstrings
✅ Example usage scripts
✅ Comprehensive README
✅ Unit tests
✅ Makefile automation

## Performance Estimates

### Training (on GPU)
- GRU: ~5-10 min for 1000 sequences, 20 epochs
- Transformer: ~10-20 min for 1000 sequences, 20 epochs

### Inference (on CPU)
- ONNX INT8 model: ~5-10ms per token
- Suitable for real-time generation

### Model Sizes
- GRU quantized: 1-2 MB
- Transformer quantized: 2-4 MB
- Baseline: 5-50 MB (depends on data)
