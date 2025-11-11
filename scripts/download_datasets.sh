#!/bin/bash
# Download and prepare training datasets
# Downloads: Lakh MIDI Clean + MAESTRO v1

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

echo "================================================"
echo "Nomad Muse Trainer - Dataset Download"
echo "================================================"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "📁 Data directory: $DATA_DIR"
echo ""

# ===================================================================
# 1. LAKH MIDI DATASET (Clean & Matched)
# ===================================================================

echo "================================================"
echo "1. Downloading Lakh MIDI Dataset (Clean)"
echo "================================================"
echo ""
echo "This dataset contains ~17,000 cleaned MIDI files"
echo "matched to the Million Song Dataset."
echo ""

LAKH_DIR="$DATA_DIR/lakh_clean"

if [ -d "$LAKH_DIR" ] && [ "$(ls -A $LAKH_DIR/*.mid 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "✓ Lakh dataset already exists ($LAKH_DIR)"
    echo "  Skipping download..."
else
    echo "Downloading Lakh MIDI Clean dataset..."
    
    # Download the clean MIDI subset
    LAKH_URL="http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
    
    echo "Downloading from: $LAKH_URL"
    echo "This may take several minutes (~2 GB)..."
    echo ""
    
    wget -c "$LAKH_URL" -O lmd_matched.tar.gz
    
    echo ""
    echo "Extracting archive..."
    tar -xzf lmd_matched.tar.gz
    
    # Organize into flat structure
    echo "Organizing files..."
    mkdir -p "$LAKH_DIR"
    find lmd_matched -name "*.mid" -exec cp {} "$LAKH_DIR/" \;
    
    # Cleanup
    rm -rf lmd_matched lmd_matched.tar.gz
    
    NUM_LAKH=$(ls "$LAKH_DIR"/*.mid 2>/dev/null | wc -l)
    echo "✓ Lakh dataset ready: $NUM_LAKH MIDI files"
fi

echo ""

# ===================================================================
# 2. MAESTRO v1 MIDI Dataset
# ===================================================================

echo "================================================"
echo "2. Downloading MAESTRO v1 Dataset"
echo "================================================"
echo ""
echo "This dataset contains ~1,000 piano performances"
echo "from international piano competitions."
echo ""

MAESTRO_DIR="$DATA_DIR/maestro_v1"

if [ -d "$MAESTRO_DIR" ] && [ "$(ls -A $MAESTRO_DIR/*.mid* 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "✓ MAESTRO dataset already exists ($MAESTRO_DIR)"
    echo "  Skipping download..."
else
    echo "Downloading MAESTRO v1 dataset..."
    
    # MAESTRO v1.0.0
    MAESTRO_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip"
    
    echo "Downloading from: $MAESTRO_URL"
    echo "This may take several minutes (~70 MB)..."
    echo ""
    
    wget -c "$MAESTRO_URL" -O maestro-v1.zip
    
    echo ""
    echo "Extracting archive..."
    unzip -q maestro-v1.zip
    
    # Organize into flat structure
    echo "Organizing files..."
    mkdir -p "$MAESTRO_DIR"
    find maestro-v1.0.0 -name "*.mid" -o -name "*.midi" | while read file; do
        cp "$file" "$MAESTRO_DIR/"
    done
    
    # Cleanup
    rm -rf maestro-v1.0.0 maestro-v1.zip
    
    NUM_MAESTRO=$(ls "$MAESTRO_DIR"/*.mid* 2>/dev/null | wc -l)
    echo "✓ MAESTRO dataset ready: $NUM_MAESTRO MIDI files"
fi

echo ""

# ===================================================================
# Summary
# ===================================================================

echo "================================================"
echo "✅ Dataset Download Complete!"
echo "================================================"
echo ""

TOTAL_LAKH=$(find "$LAKH_DIR" -name "*.mid" 2>/dev/null | wc -l)
TOTAL_MAESTRO=$(find "$MAESTRO_DIR" -name "*.mid*" 2>/dev/null | wc -l)
TOTAL_FILES=$((TOTAL_LAKH + TOTAL_MAESTRO))

echo "Dataset Summary:"
echo "  • Lakh Clean:  $TOTAL_LAKH files"
echo "  • MAESTRO v1:  $TOTAL_MAESTRO files"
echo "  • Total:       $TOTAL_FILES MIDI files"
echo ""

echo "Data directory structure:"
tree -L 2 "$DATA_DIR" 2>/dev/null || ls -lh "$DATA_DIR"

echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Index the MIDI files:"
echo "   make index"
echo "   # or: python -m scripts.index_midi --midi_dir ./data --out ./artifacts/manifest.csv"
echo ""
echo "2. Prepare the dataset:"
echo "   make prep"
echo "   # or: python -m src.data_prep --midi_dir ./data --out ./artifacts/dataset.npz"
echo ""
echo "3. Start training:"
echo "   make train"
echo "   # or: python -m src.train --dataset ./artifacts/dataset.npz --model gru --epochs 20"
echo ""
echo "Or run the complete pipeline:"
echo "   make all"
echo ""
