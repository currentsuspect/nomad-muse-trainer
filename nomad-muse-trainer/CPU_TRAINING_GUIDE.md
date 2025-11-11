# CPU Training Guide for Nomad Muse Trainer

## ✅ CPU Optimizations Applied

Your training pipeline has been optimized for **4-core / 16GB Codespaces** environment:

### 1. **PyTorch CPU Configuration**
- `torch.set_num_threads(4)` - Matches your 4-core CPU
- cuDNN disabled (not needed for CPU)
- Pin memory disabled (GPU-only feature)

### 2. **Batch Size & Memory**
- Default batch size: **12** (down from 32)
- Recommended range: 8-16 for 4-core/16GB
- Memory cleanup every 50 batches during training
- Can override with `--batch_size` flag

### 3. **DataLoader Workers**
- Default workers: **2** (down from 4)
- Prevents CPU thrashing on 4-core system
- Optimal for I/O without excessive context switching

### 4. **Codespace Safety**
- ✅ Checkpoints saved **every epoch** (auto-saves even on disconnect)
- ✅ Milestone checkpoints every 5 epochs
- ✅ Best model checkpoint saved on improvement
- Location: `artifacts/checkpoints/{model}/`

### 5. **Progress Tracking**
- Real-time progress bars with tqdm
- ETA calculation based on actual epoch times
- Memory usage tracking
- Clear epoch summaries with loss curves

## 🚀 Quick Start

### Step 1: Prepare Data
```bash
python -m src.data_prep \
  --midi_dir ./data \
  --out ./artifacts/dataset.npz
```

This will:
- Tokenize all MIDI files from `data/lakh_clean/` and `data/maestro_v1/`
- Create train/val/test splits (80/10/10)
- Save to `artifacts/dataset.npz`
- Expected time: ~10-15 minutes for 18K files

### Step 2: Train GRU Model (Recommended First)
```bash
python -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model gru \
  --epochs 20
```

**Why GRU first?**
- Smaller model (~200K params vs ~400K for Transformer)
- Faster training (~1-2 min/epoch on 4-core CPU)
- Good baseline for music generation

### Step 3: Train Transformer (Optional)
```bash
python -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model transformer \
  --epochs 20
```

**Transformer notes:**
- Larger model (~400K params)
- Slower training (~2-3 min/epoch)
- May achieve better quality

## ⚙️ Advanced Options

### Custom Batch Size
```bash
python -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model gru \
  --batch_size 8 \
  --epochs 20
```

**Batch size guide:**
- `8`: Most conservative, lowest memory
- `12`: Default, balanced (recommended)
- `16`: Higher throughput, more memory

### Resume from Checkpoint
```bash
python -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model gru \
  --resume artifacts/checkpoints/gru/latest.pt
```

### Override Model Size
```bash
# Smaller GRU (faster training)
python -m src.train \
  --dataset ./artifacts/dataset.npz \
  --model gru \
  --hidden 64 \
  --layers 1 \
  --epochs 20
```

## 📊 What to Expect

### GRU Model (Default: 128 hidden, 2 layers)
- **Parameters:** ~200,000
- **Training speed:** 1-2 min/epoch on 4-core CPU
- **20 epochs:** ~30-40 minutes total
- **Memory:** ~2-3 GB during training
- **Validation loss:** Should drop below 2.0 for decent results

### Transformer Model (Default: 256 d_model, 2 layers)
- **Parameters:** ~400,000
- **Training speed:** 2-3 min/epoch on 4-core CPU
- **20 epochs:** ~50-60 minutes total
- **Memory:** ~3-4 GB during training
- **Validation loss:** Should drop below 1.8 for good results

## 🎯 Training Progress

You'll see output like this:

```
======================================================================
🖥️  CPU-OPTIMIZED TRAINING MODE
======================================================================

Environment:
  • PyTorch threads: 4
  • Available CPU cores: 4
  • Batch size: 12 (auto-adjusted for CPU)
  • DataLoader workers: 2

Training configuration:
  • Model: gru
  • Epochs: 20
  • Learning rate: 0.001
  • Early stopping patience: 5

⚠️  CPU Training Notes:
  • No GPU detected - training on CPU only
  • Checkpoints saved every epoch for Codespace safety
  • Recommended: start with fewer epochs (10-20) for initial testing
  • Expected speed: ~1-2 min/epoch on 4-core CPU
======================================================================

======================================================================
STARTING TRAINING
======================================================================
Total epochs: 20
Early stopping patience: 5

----------------------------------------------------------------------
EPOCH 1/20
----------------------------------------------------------------------
Training: 100%|████████████| 1234/1234 [01:23<00:00, loss=3.4521]
Validation: 100%|████████████| 154/154 [00:08<00:00]

  Train loss: 3.4521
  Val loss: 2.8934
  ✓ New best! Improved by 0.5587
  💾 Checkpoint saved: latest.pt
  ⭐ Best model saved: best.pt (val_loss=2.8934)
  📌 Milestone saved: epoch_001.pt
  ⏱️  Epoch time: 91.3s | Total: 1.5min | ETA: 0:28:45

----------------------------------------------------------------------
EPOCH 2/20
----------------------------------------------------------------------
...
```

## 🛟 Codespace Disconnection Recovery

If your Codespace disconnects during training:

1. **Checkpoints are saved automatically** every epoch
2. Resume from last checkpoint:
   ```bash
   python -m src.train \
     --dataset ./artifacts/dataset.npz \
     --model gru \
     --resume artifacts/checkpoints/gru/latest.pt
   ```

3. Check checkpoint directory:
   ```bash
   ls -lh artifacts/checkpoints/gru/
   # You'll see: best.pt, latest.pt, epoch_005.pt, epoch_010.pt, etc.
   ```

## 🎵 After Training

### Export to ONNX (INT8 Quantized)
```bash
python -m src.quantize_export \
  --checkpoint artifacts/checkpoints/gru/best.pt \
  --vocab artifacts/vocab.json \
  --out artifacts/nomad_muse_gru_int8.onnx
```

### Generate Sample Music
```bash
python scripts/demo_sample.py \
  --model artifacts/nomad_muse_gru_int8.onnx \
  --vocab artifacts/vocab.json \
  --out sample_output.mid \
  --length 512
```

## 🐛 Troubleshooting

### "Out of memory" error
- Reduce batch size: `--batch_size 8`
- Use smaller model: `--hidden 64 --layers 1`

### Training too slow
- Start with fewer epochs: `--epochs 10`
- Train on subset by reducing sequence_length in config.yaml
- Consider smaller model architecture

### Loss not decreasing
- Train for more epochs (50+)
- Increase model size: `--hidden 256`
- Check dataset quality with data_prep output logs

## 📝 Configuration Files

### config.yaml (CPU-Optimized)
- `batch_size: 12` (was 32)
- `num_workers: 2` (was 4)
- All other settings optimized for CPU training

### Modify config.yaml for different runs
```yaml
training:
  batch_size: 12      # Adjust 8-16 based on memory
  epochs: 50          # Increase for better quality
  learning_rate: 0.001
  early_stop_patience: 5
  num_workers: 2      # Keep at 2 for 4-core CPU
```

## 🎯 Recommended Training Workflow

1. **Quick test run (10 min):**
   ```bash
   python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 5
   ```

2. **Standard training (30-40 min):**
   ```bash
   python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20
   ```

3. **High-quality training (2-3 hours):**
   ```bash
   python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 50
   ```

4. **Export and test:**
   ```bash
   python -m src.quantize_export \
     --checkpoint artifacts/checkpoints/gru/best.pt \
     --vocab artifacts/vocab.json \
     --out artifacts/nomad_muse_gru_int8.onnx
   ```

---

**Ready to train!** Start with the Quick Start section above. 🚀
