.PHONY: help install prep train export eval demo clean

help:
	@echo "nomad-muse-trainer - Tiny MIDI music model trainer"
	@echo ""
	@echo "Usage:"
	@echo "  make install    - Install dependencies"
	@echo "  make download   - Download Lakh + MAESTRO datasets"
	@echo "  make index      - Index MIDI files in ./data"
	@echo "  make prep       - Prepare dataset from MIDI files"
	@echo "  make train      - Train model (default: GRU)"
	@echo "  make train-tx   - Train transformer model"
	@echo "  make export     - Quantize and export to ONNX"
	@echo "  make baseline   - Build and export baseline model"
	@echo "  make eval       - Evaluate model on test set"
	@echo "  make demo       - Generate sample MIDI with ONNX model"
	@echo "  make all        - Run full pipeline (prep → train → export → eval)"
	@echo "  make clean      - Remove artifacts"

install:
	pip install -r requirements.txt

download:
	python -m scripts.download_data

index:
	python -m scripts.index_midi --midi_dir ./data --out ./artifacts/manifest.csv

prep:
	python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz

train:
	python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20

train-tx:
	python -m src.train --dataset ./artifacts/dataset.npz --model transformer --epochs 20

export:
	python -m src.quantize_export --ckpt ./artifacts/checkpoints/best.pt --out ./artifacts/muse_quantized.onnx

baseline:
	python -m src.baseline.export --dataset ./artifacts/dataset.npz --out ./artifacts/baseline.bin

eval:
	python -m src.evaluate --dataset ./artifacts/dataset.npz --onnx ./artifacts/muse_quantized.onnx

demo:
	python -m scripts.demo_sample --onnx ./artifacts/muse_quantized.onnx --out ./artifacts/sample.mid

all: prep train export baseline eval demo

clean:
	rm -rf artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
