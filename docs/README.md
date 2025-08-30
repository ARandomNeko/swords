# ⚔️ PVC Sword Fighting Game

A real-time sword fighting game using computer vision and MediaPipe pose detection.

## 🎯 Overview

This project creates an interactive sword fighting game where players use PVC pipes as swords. The system uses computer vision to detect player movements and award points for successful hits. Perfect for hackathons, demos, and fun competitions!

## ✨ Features

- **Real-time pose detection** with MediaPipe
- **2-player simultaneous gameplay**
- **Hit detection** with visual feedback
- **RGB camera auto-detection** (avoids IR cameras)
- **High resolution support** (up to 1920x1080)
- **Interactive controls** and live scoreboard
- **Comprehensive documentation**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (RGB camera recommended)
- 2 PVC pipes for swords
- Clear play area with good lighting

### Installation (Nix - Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd swords

# Enable the Nix environment
direnv allow  # if using direnv
# or
nix develop
```

### Installation (Manual)
```bash
pip install mediapipe opencv-python numpy
```

### Run the Demo
```bash
# Test pose detection
python pose_extractor.py

# Play the full game
python game.py
```

## 📚 Documentation

Comprehensive guides are available in the `docs/` folder:

- **[Pose Extractor Guide](docs/pose_extractor_guide.md)** - Complete API reference and usage
- **[Game Demo Guide](docs/game_demo_guide.md)** - How to play and run the game
- **[Sword Fighting Game Design](docs/Sword_Fighting_Game.md)** - Original design document
- **[Report Template](docs/Sword_Fighting_Game_Report_Template.md)** - Hackathon report template

## 🎮 How to Play

1. **Setup**: Position camera 6-8 feet away, ensure good lighting
2. **Players**: Stand on opposite sides of the center line
3. **Objective**: Hit your opponent's head or torso with your sword (wrist tracking)
4. **Scoring**: First to 3 points wins (or highest score after 30 seconds)
5. **Controls**: Use physical movement - no keyboard controls needed!

### Game Controls
- **'b'**: Toggle bounding boxes (pose extractor demo)
- **'ESC'**: Exit application

## 🏗️ Project Structure

```
swords/
├── pose_extractor.py          # Main pose detection module
├── game.py                    # Full sword fighting game
├── flake.nix                  # Nix development environment
├── docs/                      # Documentation folder
│   ├── README.md              # This file
│   ├── pose_extractor_guide.md # Pose extractor documentation
│   ├── game_demo_guide.md     # Game guide
│   └── ...                    # Other docs
```

## 🔧 Configuration

### Pose Detection Settings
```python
# In pose_extractor.py
extractor = PoseExtractor(
    min_detection_confidence=0.5,  # Adjust sensitivity
    min_tracking_confidence=0.5,   # Adjust tracking
    model_complexity=1,            # 0=fast, 1=balanced, 2=accurate
    max_num_poses=2                # Number of players
)
```

### Game Settings
```python
# In game.py
WINNING_SCORE = 3        # Points needed to win
GAME_DURATION = 30       # Game time in seconds
HIT_COOLDOWN = 0.5       # Cooldown between hits
```

## 🎯 Use Cases

### Hackathons
- **Wildcard Innovation Track**: Perfect for creative/fun categories
- **Computer Vision Demos**: Showcases real-time CV applications
- **Interactive Exhibits**: Engaging audience participation

### Educational
- **Computer Vision Learning**: Hands-on MediaPipe experience
- **Game Development**: Real-time interaction and state management
- **Physical Computing**: Bridging digital and physical worlds

### Entertainment
- **Party Games**: Fun multiplayer activity
- **Competitions**: Tournament-style gameplay
- **Team Building**: Interactive group activity

## 🔬 Technical Details

### Computer Vision
- **MediaPipe Pose**: Google's pose detection solution
- **Real-time Processing**: 30+ FPS on modern hardware
- **Multi-person Detection**: Region-based approach for 2 players
- **Hit Detection**: Bounding box collision detection

### Architecture
- **Modular Design**: Separate pose detection and game logic
- **Clean API**: Easy to integrate and extend
- **Error Handling**: Robust camera and detection error handling
- **Performance Optimized**: Efficient processing pipeline

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure camera is connected and not in use
- Try different USB ports
- Check camera permissions

**Poor pose detection:**
- Improve lighting conditions
- Ensure players are fully visible
- Adjust detection confidence settings

**Only one player detected:**
- Stand on opposite sides of center line
- Increase distance between players
- Ensure both sides are well-lit

For detailed troubleshooting, see the [documentation](docs/).

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Color-based sword detection**: Using colored tape on sword tips
- **Additional game modes**: Different rules and challenges
- **Performance optimization**: Faster processing
- **UI improvements**: Better visual feedback
- **Multi-platform support**: Testing on different systems

## 📄 License

This project is open source. See LICENSE file for details.

## 🏆 Hackathon Usage

### Enigma Hackathon 2025 - Wildcard Innovation Track

This project was designed for the Wildcard Innovation track, demonstrating:
- **Creative use of technology**: Computer vision for gaming
- **Real-time interaction**: Immediate visual feedback
- **Physical engagement**: Moving beyond traditional interfaces
- **Accessible fun**: Easy to understand and play

### Demo Script (5 minutes)
1. **Show pose detection** (1 min): Single person with bounding boxes
2. **Demonstrate 2-player** (1 min): Two people, show split detection
3. **Play quick match** (3 min): Live gameplay with scoring

### Judging Points
- **Technical innovation**: Real-time multi-person pose detection
- **User experience**: Intuitive physical interaction
- **Entertainment value**: Fun and engaging gameplay
- **Practical implementation**: Working demo with clear rules

## 📞 Support

- **Documentation**: Check the `docs/` folder for detailed guides
- **Issues**: Use the issue tracker for bug reports
- **Questions**: See troubleshooting sections in documentation

---

**Have fun sword fighting!** ⚔️🎮

*Built with ❤️ for interactive computer vision gaming*