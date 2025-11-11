# Artifacts Directory

This directory will be automatically created to store:

- Processed datasets (`.npz` files)
- Trained model checkpoints (`.pt` files)
- Exported ONNX models (`.onnx` files)
- Vocabulary files (`vocab.json`)
- Statistics (`stats.json`)
- Baseline models (`baseline.bin`)
- Generated samples (`.mid` files)

**Note**: This directory is excluded from git (see `.gitignore`).

## Structure After Running Pipeline

```
artifacts/
├── dataset.npz              # Tokenized sequences (train/val/test)
├── vocab.json               # Token vocabulary mapping
├── stats.json               # Dataset and model statistics
├── manifest.csv             # MIDI file index (if using index_midi.py)
├── muse_quantized.onnx      # Exported quantized model
├── baseline.bin             # Baseline Markov model
├── sample.mid               # Generated sample
├── sample_eval.mid          # Evaluation sample
└── checkpoints/
    ├── best.pt              # Best model checkpoint
    └── latest.pt            # Latest checkpoint
```

## Typical Sizes

- `dataset.npz`: 10-500 MB (depends on number of MIDI files)
- `vocab.json`: < 1 MB
- `*.pt` checkpoints: 1-10 MB
- `*.onnx` model: 1-5 MB (quantized)
- `baseline.bin`: 5-50 MB
