# License Quick Reference

## Nomad Muse Trainer Dual Licensing

### 📂 What's Open Source (MIT License)

✅ **YOU CAN USE FREELY:**
- All Python source code
- Training scripts and pipeline
- Configuration examples
- Documentation
- Build tools (Makefile, setup.py)

### 🔒 What's Proprietary (All Rights Reserved)

❌ **YOU CANNOT REDISTRIBUTE:**
- Trained model weights (.pt, .onnx files)
- Datasets and MIDI training data
- Generated artifacts from proprietary training
- "Nomad Muse" and "Nomad Studios" trademarks

---

## Common Scenarios

### ✅ Allowed

**Q: Can I use this code to train my own music models?**  
A: Yes! Train on your own data and own the results.

**Q: Can I modify the code?**  
A: Yes! Fork, modify, and distribute your changes under MIT.

**Q: Can I use this commercially?**  
A: Yes! Use the code in commercial products.

**Q: Can I share the source code?**  
A: Yes! Just include the LICENSE and NOTICE files.

### ❌ Not Allowed (Without Permission)

**Q: Can I download and redistribute pre-trained Nomad Muse models?**  
A: No. Those are proprietary to Nomad Studios.

**Q: Can I use the "Nomad Muse" name for my product?**  
A: No. That's a trademark of Nomad Studios.

**Q: Can I use Nomad Studios' training datasets?**  
A: No. Those are proprietary unless explicitly shared.

**Q: Can I extract weights from published models?**  
A: No. Reverse engineering proprietary models is prohibited.

---

## Files and Their Licenses

| Path | License | Notes |
|------|---------|-------|
| `src/*.py` | MIT | Open source code |
| `scripts/*.py` | MIT | Open source utilities |
| `config.yaml` | MIT | Example configuration |
| `Makefile` | MIT | Build automation |
| `README.md` | MIT | Documentation |
| `artifacts/*.onnx` | Proprietary | If trained with Nomad data |
| `artifacts/*.pt` | Proprietary | If trained with Nomad data |
| `data/*.mid` | Varies | Check your data source |
| `artifacts/vocab.json` | Proprietary* | If from Nomad training |

\* If you generate vocab.json from your own data, you own it.

---

## Need Permission For:

- Commercial redistribution of trained models
- Using "Nomad Muse" or "Nomad Studios" in product names
- Bundling proprietary models with your products
- Using proprietary datasets

**Contact:** licensing@nomadstudios.example.com

---

## Complete License Documents

- **LICENSE** - Full MIT License text
- **NOTICE** - Detailed proprietary asset terms
- **README.md** - License section with examples

---

**Last Updated:** October 27, 2025  
**Copyright © 2025 Nomad Studios**
