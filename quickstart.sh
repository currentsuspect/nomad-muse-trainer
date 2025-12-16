#!/bin/bash
# Quick Start Script for Multi-Expert Nomad Muse Trainer

echo "🎵 Multi-Expert Nomad Muse Trainer - Quick Start"
echo "=================================================="

# Create data directory if it doesn't exist
mkdir -p data/raw

# Check if we have MIDI files
midi_count=$(find data/raw -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l)

if [ "$midi_count" -eq 0 ]; then
    echo "📁 No MIDI files found in data/raw/"
    echo "📝 Adding sample MIDI files for testing..."
    
    # Create a simple test
    python test_simple.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Sample test completed successfully!"
    else
        echo "❌ Sample test failed. Please check your installation."
        exit 1
    fi
else
    echo "✅ Found $midi_count MIDI files in data/raw/"
fi

echo ""
echo "🚀 Running Multi-Expert End-to-End Test..."
python test_multi_expert.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Success! Your Multi-Expert system is working!"
    echo ""
    echo "Next steps:"
    echo "1. Add your MIDI files to data/raw/"
    echo "2. Train models: python src/multi_expert_train.py --midi_dir data/raw --expert all"
    echo "3. Evaluate: python src/multi_expert_train.py --expert all --eval_only"
    echo ""
    echo "For detailed documentation, see:"
    echo "- README.md (main system)"
    echo "- MULTI_EXPERT_README.md (multi-expert details)"
else
    echo ""
    echo "❌ Multi-expert test failed. Check the output above for details."
    echo ""
    echo "Troubleshooting:"
    echo "1. Run setup: ./setup.sh"
    echo "2. Check dependencies: pip list"
    echo "3. Run simple test: python test_simple.py"
    exit 1
fi
