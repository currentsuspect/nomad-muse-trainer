# Multi-Expert Nomad Muse Trainer

## Overview

This document describes the **hard route** implementation of Nomad Muse Trainer - a multi-expert system designed for production-ready, real-time music generation in C++ DAWs.

## Architecture Overview

The system implements **four specialized experts** working in coordination:

1. **MuseDrums**: Percussion generation (groove, fills, variations)
2. **MuseHarmony**: Chord progression generation (stable progressions, key awareness)  
3. **MuseMelody**: Melodic generation (conditioned on harmony)
4. **MuseConductor**: Arrangement control (future expansion)

### Design Principles

- **Separation of Concerns**: Each expert handles one musical domain
- **Real-time Ready**: All models optimized for CPU inference
- **Reproducible**: Every training run is fingerprinted
- **Musically Grounded**: Tokenization preserves musical meaning
- **Evaluation-First**: Musical quality metrics beyond perplexity

## Canonical Nomad Symbolic Format

The foundation of the system is a **canonical intermediate representation** that all tokenizers derive from:

```json
{
    "global": {
        "tempo_map": [(time_seconds, bpm), ...],
        "time_signature_map": [(time_seconds, (num, den)), ...],
        "ppq": 480,
        "duration_seconds": 120.0
    },
    "tracks": [
        {
            "id": 0,
            "name": "Drums",
            "is_drum": true,
            "classification": {
                "type": "drums",
                "confidence": 0.95,
                "reasons": ["GM percussion channel"]
            },
            "events": [...]
        }
    ],
    "chords": [
        {
            "time_start": 0.0,
            "bar_start": 0,
            "chord": {
                "root": "C",
                "quality": "maj",
                "confidence": 0.8
            }
        }
    ],
    "metadata": {
        "source_file": "song.mid",
        "quantization_grid": "16th"
    }
}
```

## Expert Tokenizers

### MuseDrums Tokenizer

**Purpose**: Generate drum patterns, grooves, and fills

**Vocabulary**:
- `BAR_<n>`, `BEAT_<n>`, `SUBSTEP_<n>`: Musical position
- `DRUM_HIT_<pitch>`: GM drum pitch events (35-81)
- `VEL_BIN_<k>`: Velocity quantization (8 bins)
- `REST_<duration>`: Timing control
- `FILL_START`, `FILL_END`: Fill generation markers

**Key Features**:
- Groove continuation and pattern variation
- Support for fills and variations
- Musically meaningful timing (16th note grid)
- Small vocabulary for fast inference

### MuseHarmony Tokenizer

**Purpose**: Generate stable chord progressions

**Vocabulary**:
- `BAR_<n>`, `BEAT_<n>`: Bar/beat markers
- `KEY_<root>_<mode>`: Key signatures (major/minor)
- `CHORD_<root>_<quality>`: Chord tokens (maj, min, dom7, etc.)
- `HOLD_<beats>`: Harmonic rhythm control
- `REST`: No chord periods

**Key Features**:
- Key-aware chord generation
- Harmonic rhythm modeling
- Voice leading considerations
- Stable progression patterns

### MuseMelody Tokenizer

**Purpose**: Generate melodies and basslines (conditioned on harmony)

**Vocabulary**:
- `BAR_<n>`, `BEAT_<n>`, `SUBSTEP_<n>`: Position markers
- `NOTE_<pitch>`: Note events (MIDI pitches 0-127)
- `VEL_BIN_<k>`, `DUR_BIN_<k>`: Velocity and duration
- `CONDITION_CHORD_<root>_<quality>`: Harmony conditioning
- `CONDITION_KEY_<root>_<mode>`: Key conditioning

**Key Features**:
- Harmony conditioning support
- Chord-tone relationship modeling
- Phrase coherence
- Contour analysis

## Model Architectures

### v1: GRU Fast Start (Current)

All experts use **GRU-based architectures** optimized for their domain:

#### MuseDrums (Small, Fast)
- Hidden size: 64
- Layers: 2
- Multi-head output: groove consistency + variation
- Parameters: ~50K

#### MuseHarmony (Medium)
- Hidden size: 96  
- Layers: 3
- Key/chord embeddings
- Parameters: ~100K

#### MuseMelody (Largest)
- Hidden size: 128
- Layers: 3
- Harmony conditioning
- Parameters: ~150K

### Future v2: Transformer Upgrade
- Replace GRU with lightweight Transformer
- Maintain domain-specific heads
- Better long-range dependencies

### Future v3: Distillation
- Train larger teacher models
- Distill to tiny student models
- Maximum efficiency for DAW integration

## Training Pipeline

### 1. Dataset Preparation
```bash
# Convert MIDI files to Nomad format
python -m src.multi_expert_train \
  --midi_dir ./data \
  --out ./artifacts \
  --prepare_only
```

### 2. Expert Training
```bash
# Train all experts
python -m src.multi_expert_train \
  --midi_dir ./data \
  --config config_multi_expert.yaml \
  --expert all \
  --epochs 50

# Train individual expert
python -m src.multi_expert_train \
  --expert drums \
  --epochs 20
```

### 3. Evaluation
```bash
# Musical evaluation (beyond perplexity)
python -m src.multi_expert_train \
  --expert all \
  --eval_only
```

## Musical Evaluation Metrics

### Beyond Perplexity: Muse Student Report Card

#### Drum Metrics
- **Groove Consistency**: Pattern similarity across bars
- **Hit Density**: Distribution of drum hits
- **Repetition vs Variation**: Balance of patterns

#### Harmony Metrics  
- **Progression Plausibility**: Comparison to common progressions
- **Key Consistency**: Harmonic coherence
- **Harmonic Rhythm**: Chord change patterns

#### Melody Metrics
- **Chord-Tone Ratio**: % of melody notes in current chord
- **Melodic Contour**: Interval statistics and patterns
- **Phrase Coherence**: Motif repetition and variation

#### Memorization Detection
- **N-gram Analysis**: Detect copied sequences
- **Shingle Similarity**: Pattern matching vs training data
- **Diversity Metrics**: Ensure generative diversity

## Configuration

### Multi-Expert Config (`config_multi_expert.yaml`)

```yaml
# Expert model configurations
drums_model:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2

harmony_model:
  hidden_size: 96
  num_layers: 3
  dropout: 0.1

melody_model:
  hidden_size: 128
  num_layers: 3
  dropout: 0.2

# Expert-specific settings
drums:
  time_signature: [4, 4]
  max_bars: 16

harmony:
  include_keys: true

melody:
  conditioning: "chord"  # none, chord, key, both
```

## End-to-End Testing

Run the complete test suite:

```bash
python test_multi_expert.py
```

This test:
1. Creates sample MIDI files with drums, harmony, melody
2. Converts to Nomad Symbolic Format
3. Tests all expert tokenizers
4. Trains each expert (reduced epochs for testing)
5. Evaluates musical quality metrics
6. Reports overall system health

## Production Deployment

### ONNX Export
```bash
# Export each expert to ONNX
python -m src.quantize_export \
  --ckpt ./artifacts/checkpoints/drums_best.pt \
  --out ./artifacts/drums_quantized.onnx

python -m src.quantize_export \
  --ckpt ./artifacts/checkpoints/harmony_best.pt \
  --out ./artifacts/harmony_quantized.onnx

python -m src.quantize_export \
  --ckpt ./artifacts/checkpoints/melody_best.pt \
  --out ./artifacts/melody_quantized.onnx
```

### C++ Integration Example
```cpp
// Load expert models
Ort::Session drums_session(env, "drums_quantized.onnx", options);
Ort::Session harmony_session(env, "harmony_quantized.onnx", options);
Ort::Session melody_session(env, "melody_quantized.onnx", options);

// Generation loop
std::vector<int> drum_tokens = {BOS_TOKEN};
std::vector<int> harmony_tokens = {BOS_TOKEN};
std::vector = {BOS<int> melody_tokens_TOKEN};

for (int bar = 0; bar < 16; bar++) {
    // Generate drum pattern
    auto drum_output = generate_next(drums_session, drum_tokens);
    drum_tokens.push_back(drum_output);
    
    // Generate chord progression
    auto harmony_output = generate_next(harmony_session, harmony_tokens);
    harmony_tokens.push_back(harmony_output);
    
    // Generate melody (conditioned on harmony)
    auto melody_output = generate_next(melody_session, melody_tokens, harmony_tokens);
    melody_tokens.push_back(melody_output);
}
```

## Performance Characteristics

### Model Sizes (Estimated)
- **MuseDrums**: ~50K parameters → ~200KB ONNX
- **MuseHarmony**: ~100K parameters → ~400KB ONNX  
- **MuseMelody**: ~150K parameters → ~600KB ONNX
- **Total**: ~300K parameters → ~1.2MB ONNX

### Inference Speed
- **CPU Target**: <1ms per token on modern CPU
- **Memory**: <10MB total RAM usage
- **Latency**: Real-time generation capability

## Risks and Mitigations

### Data Quality Risks
- **Risk**: Poor MIDI files, inconsistent quantization
- **Mitigation**: Robust preprocessing, validation pipeline
- **Monitoring**: Track data quality metrics during training

### Chord Labeling Errors
- **Risk**: Incorrect chord extraction leads to bad harmony models
- **Mitigation**: Confidence thresholds, multiple algorithms
- **Validation**: Manual inspection of chord extraction samples

### Genre Bias
- **Risk**: Models only work well on training genres
- **Mitigation**: Diverse dataset, augmentation strategies
- **Testing**: Cross-genre evaluation metrics

### Overfitting/Memorization
- **Risk**: Models memorize training data
- **Mitigation**: Memorization detection, regularization
- **Monitoring**: N-gram similarity analysis

## Future Extensions

### Phase 2: Conditioned Melody
- Harmony conditioning for MuseMelody
- Advanced musical evaluation metrics
- Multi-expert inference coordination

### Phase 3: Production Ready
- Teacher-student distillation
- Advanced quantization (static)
- C++ integration harness
- Real-time streaming interface

## Contributing

### Adding New Experts
1. Create expert tokenizer in `src/tokenizers_<expert>.py`
2. Implement expert model in `src/expert_models.py`
3. Add training pipeline support
4. Update evaluation metrics
5. Test end-to-end integration

### Modifying Tokenization
1. Update canonical format if needed
2. Modify expert tokenizers consistently
3. Retrain all affected experts
4. Update evaluation metrics
5. Verify backwards compatibility

## License

This multi-expert implementation follows the same licensing as the base Nomad Muse Trainer:
- **Source Code**: MIT License
- **Trained Models**: All Rights Reserved
- **Proprietary Assets**: Nomad Studios

For commercial licensing of multi-expert models: licensing@nomadstudios.example.com