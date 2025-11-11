"""Setup script for Nomad Muse Trainer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path("README.md").read_text(encoding="utf-8")

setup(
    name="nomad-muse-trainer",
    version="0.1.0",
    description="Train tiny CPU-friendly music models from MIDI for DAW inference",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Nomad Studios",
    author_email="contact@nomadstudios.example.com",
    url="https://github.com/nomadstudios/nomad-muse-trainer",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pretty_midi>=0.2.10",
        "miditoolkit>=1.0.1",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="music midi machine-learning onnx daw generation nomad-muse",
    license="MIT (source code only; see NOTICE for proprietary assets)",
)
