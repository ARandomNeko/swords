# ⚔️ PVC Sword Fighting Game

A real-time sword fighting game using computer vision and MediaPipe pose detection.

## 🚀 Quick Start

```bash
# Test pose detection (2-person support)
python pose_extractor.py

# Play the full game
python game.py
```

## 🎯 Features

- **Real-time 2-person pose detection**
- **RGB camera auto-detection** (1920x1080 support)
- **Hit detection with visual feedback**
- **Interactive controls and live scoreboard**

## 📚 Documentation

Complete guides available in the `docs/` folder:

- **[📖 Main Documentation](docs/README.md)** - Complete overview
- **[🎯 Pose Extractor Guide](docs/pose_extractor_guide.md)** - API reference and usage
- **[🎮 Game Demo Guide](docs/game_demo_guide.md)** - How to play and setup
- **[⚔️ Original Design](docs/Sword_Fighting_Game.md)** - Game concept and design

## 🎮 How to Play

1. Stand on opposite sides of the camera view
2. Use PVC pipes as swords (system tracks your wrists)
3. Hit opponent's head or torso for points
4. First to 3 points wins!

### Controls
- **'b'**: Toggle bounding boxes
- **'ESC'**: Exit

## 🔧 Environment Setup

Using Nix (recommended):
```bash
direnv allow  # Auto-loads environment
```

Manual setup:
```bash
pip install mediapipe opencv-python numpy
```

---

**For complete documentation, see [docs/README.md](docs/README.md)**
