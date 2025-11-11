"""Download training datasets: Lakh MIDI Clean and MAESTRO v1.

This script downloads and organizes datasets for training Nomad Muse models.
"""

import argparse
import os
import tarfile
import zipfile
from pathlib import Path
import urllib.request
import shutil
import sys

def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress indicator."""
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Destination: {output_path}")
    print()
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            bar_length = 50
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '-' * (bar_length - filled)
            size_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            print(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress bar
        print(f"✓ Download complete: {output_path.name}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_lakh_clean(data_dir: Path):
    """Download and extract Lakh MIDI Clean dataset."""
    print("=" * 70)
    print("1. LAKH MIDI DATASET (Clean & Matched)")
    print("=" * 70)
    print()
    print("This dataset contains ~17,000 cleaned MIDI files")
    print("matched to the Million Song Dataset.")
    print()
    
    lakh_dir = data_dir / "lakh_clean"
    
    # Check if already exists
    if lakh_dir.exists() and len(list(lakh_dir.glob("*.mid"))) > 100:
        print(f"✓ Lakh dataset already exists: {lakh_dir}")
        print(f"  Found {len(list(lakh_dir.glob('*.mid')))} MIDI files")
        print("  Skipping download...")
        return True
    
    # Download
    url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
    archive_path = data_dir / "lmd_matched.tar.gz"
    
    print("Downloading Lakh MIDI Clean dataset...")
    print("Size: ~2 GB (this may take several minutes)")
    print()
    
    if not download_file(url, archive_path, "Lakh MIDI Dataset"):
        return False
    
    # Extract
    print()
    print("Extracting archive...")
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("✓ Extraction complete")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    
    # Organize files
    print("Organizing files into flat structure...")
    lakh_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_dir = data_dir / "lmd_matched"
    midi_files = list(extracted_dir.rglob("*.mid"))
    
    for i, midi_file in enumerate(midi_files, 1):
        # Create unique filename to avoid collisions
        new_name = f"lakh_{i:05d}_{midi_file.name}"
        shutil.copy2(midi_file, lakh_dir / new_name)
        if i % 1000 == 0:
            print(f"  Copied {i}/{len(midi_files)} files...")
    
    print(f"✓ Organized {len(midi_files)} MIDI files")
    
    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(extracted_dir)
    archive_path.unlink()
    print("✓ Cleanup complete")
    
    return True


def download_maestro_v1(data_dir: Path):
    """Download and extract MAESTRO v1 dataset."""
    print()
    print("=" * 70)
    print("2. MAESTRO v1 MIDI DATASET")
    print("=" * 70)
    print()
    print("This dataset contains ~1,000 piano performances")
    print("from international piano competitions.")
    print()
    
    maestro_dir = data_dir / "maestro_v1"
    
    # Check if already exists
    if maestro_dir.exists() and len(list(maestro_dir.glob("*.mid*"))) > 100:
        print(f"✓ MAESTRO dataset already exists: {maestro_dir}")
        print(f"  Found {len(list(maestro_dir.glob('*.mid*')))} MIDI files")
        print("  Skipping download...")
        return True
    
    # Download
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip"
    archive_path = data_dir / "maestro-v1.zip"
    
    print("Downloading MAESTRO v1 dataset...")
    print("Size: ~70 MB")
    print()
    
    if not download_file(url, archive_path, "MAESTRO v1 Dataset"):
        return False
    
    # Extract
    print()
    print("Extracting archive...")
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Extraction complete")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False
    
    # Organize files
    print("Organizing files into flat structure...")
    maestro_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_dir = data_dir / "maestro-v1.0.0"
    midi_files = list(extracted_dir.rglob("*.mid")) + list(extracted_dir.rglob("*.midi"))
    
    for i, midi_file in enumerate(midi_files, 1):
        # Create unique filename
        new_name = f"maestro_{i:04d}_{midi_file.name}"
        shutil.copy2(midi_file, maestro_dir / new_name)
    
    print(f"✓ Organized {len(midi_files)} MIDI files")
    
    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(extracted_dir)
    archive_path.unlink()
    print("✓ Cleanup complete")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Lakh MIDI Clean and MAESTRO v1 datasets"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory to store datasets (default: ./data)"
    )
    parser.add_argument(
        "--lakh_only",
        action="store_true",
        help="Download only Lakh dataset"
    )
    parser.add_argument(
        "--maestro_only",
        action="store_true",
        help="Download only MAESTRO dataset"
    )
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("NOMAD MUSE TRAINER - Dataset Download")
    print("=" * 70)
    print()
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Data directory: {args.data_dir.absolute()}")
    print()
    
    success = True
    
    # Download datasets
    if not args.maestro_only:
        if not download_lakh_clean(args.data_dir):
            success = False
            print("⚠️  Lakh download failed, continuing with MAESTRO...")
    
    if not args.lakh_only:
        if not download_maestro_v1(args.data_dir):
            success = False
            print("⚠️  MAESTRO download failed")
    
    # Summary
    print()
    print("=" * 70)
    if success:
        print("✅ DATASET DOWNLOAD COMPLETE!")
    else:
        print("⚠️  DOWNLOAD COMPLETED WITH ERRORS")
    print("=" * 70)
    print()
    
    # Count files
    lakh_count = len(list((args.data_dir / "lakh_clean").glob("*.mid"))) if (args.data_dir / "lakh_clean").exists() else 0
    maestro_count = len(list((args.data_dir / "maestro_v1").glob("*.mid*"))) if (args.data_dir / "maestro_v1").exists() else 0
    total_count = lakh_count + maestro_count
    
    print("Dataset Summary:")
    print(f"  • Lakh Clean:  {lakh_count:,} files")
    print(f"  • MAESTRO v1:  {maestro_count:,} files")
    print(f"  • Total:       {total_count:,} MIDI files")
    print()
    
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print()
    print("1. Index the MIDI files:")
    print("   python -m scripts.index_midi --midi_dir ./data --out ./artifacts/manifest.csv")
    print()
    print("2. Prepare the dataset:")
    print("   python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz")
    print()
    print("3. Start training:")
    print("   python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20")
    print()
    print("Or use Makefile shortcuts:")
    print("   make index && make prep && make train")
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
