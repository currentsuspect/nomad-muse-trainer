# Nomad Muse Trainer: GPU Training Guide

This guide provides instructions for setting up your environment to train the Nomad Muse Trainer on a GPU. Training on a GPU can significantly speed up the process, allowing you to use larger models and batch sizes.

## 1. Prerequisites

- A CUDA-enabled NVIDIA GPU.
- NVIDIA drivers and the CUDA Toolkit installed on your system. You can check if CUDA is installed by running `nvcc --version`.

## 2. Install PyTorch with CUDA Support

The `requirements.txt` file is configured for CPU-only training. To use a GPU, you need to install a version of PyTorch that is compiled with CUDA support.

1.  **Uninstall the existing CPU-only PyTorch:**
    ```bash
    pip uninstall torch torchvision torchaudio
    ```

2.  **Install PyTorch with CUDA:**
    Go to the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the options that match your system (e.g., `Stable`, `Linux`, `Pip`, `Python`, and your CUDA version). This will generate an installation command.

    For example, for CUDA 11.8, the command is:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *Note: Make sure to use the command that corresponds to your specific CUDA version.*

## 3. Configure `config.yaml` for GPU Training

The `config.yaml` file allows you to control various training parameters. For GPU training, you should update the following settings:

-   **`device`**: Set this to `"cuda"` to enable GPU training.
-   **`batch_size`**: You can significantly increase the batch size when using a GPU. A good starting point is `64` or `128`, but this will depend on your GPU's VRAM.
-   **`num_workers`**: This can be increased to take advantage of more powerful CPUs for data loading. A good starting point is `4` or `8`.

Here is an example of a `training` section in `config.yaml` configured for GPU training:

```yaml
training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  early_stop_patience: 5
  checkpoint_dir: "artifacts/checkpoints"
  device: "cuda"  # Use the GPU
  num_workers: 4
```

## 4. Run the Training

Once you have installed the correct PyTorch version and updated your `config.yaml`, you can run the training script as usual:

```bash
python -m src.train --dataset artifacts/dataset.npz --model gru --config config.yaml
```

The script will now use the GPU for training, and you should see a significant improvement in training speed.
