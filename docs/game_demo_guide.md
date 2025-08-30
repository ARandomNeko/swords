# ⚔️ Sword Fighting Game Demo Guide

Complete guide to running and playing the sword fighting game demo.

## 📋 Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Running the Demo](#running-the-demo)
- [Game Rules](#game-rules)
- [Controls](#controls)
- [Gameplay Tips](#gameplay-tips)
- [Troubleshooting](#troubleshooting)

## 🔹 Overview

The sword fighting game is a real-time motion-controlled game where players use PVC pipes as swords. The system detects player movements using computer vision and awards points for successful hits.

### Game Features
- **Real-time pose detection** using MediaPipe
- **2-player simultaneous gameplay**
- **Hit detection** with different target zones (head, torso)
- **Live scoreboard** and winner announcement
- **Visual feedback** for hits and player positions

## 🚀 Setup

### Hardware Requirements
- **Webcam**: RGB camera (1080p recommended)
- **PVC Pipes**: 2 lightweight PVC pipes as swords (3-4 feet long)
- **Colored Tape**: Red and blue tape for sword tip marking (optional)
- **Space**: Clear area 10x10 feet minimum
- **Lighting**: Good even lighting

### Software Requirements
- Python 3.8+
- All dependencies installed (handled by Nix environment)

### Physical Setup
1. **Camera Position**: Mount camera 6-8 feet away at chest height
2. **Play Area**: Clear space with good lighting
3. **Sword Preparation**: 
   - Use lightweight PVC pipes
   - Optional: Wrap tips with colored tape (red/blue)
   - Ensure safe rounded ends

## 🎮 Running the Demo

### Basic Demo
```bash
# Run the basic pose extractor (for testing)
python pose_extractor.py
```

### Full Game Demo
```bash
# Run the complete sword fighting game
python game.py
```

### Demo Options
```bash
# Specify camera index
python pose_extractor.py  # Auto-detects RGB camera
python game.py           # Uses same camera detection
```

## 🎯 Game Rules

### Objective
Score points by hitting your opponent with your sword while avoiding their attacks.

### Scoring System
- **Head Hit**: +1 point (yellow bounding box)
- **Torso Hit**: +1 point (magenta bounding box)
- **Hit Cooldown**: 500ms between hits to prevent spam

### Win Conditions
- **First to 3 points** wins
- **Time limit**: 30 seconds (highest score wins)
- **Draw**: If tied after time limit

### Player Assignment
- **Player 1 (Green)**: Left side of screen
- **Player 2 (Red)**: Right side of screen
- Players are assigned automatically based on position

## 🎮 Controls

### Pose Extractor Demo
- **'b'**: Toggle bounding boxes on/off
- **'ESC'**: Exit application

### Game Demo
- **Physical Movement**: Use your body and sword to play
- **'ESC'**: Exit game
- **'r'**: Restart game (if implemented)

## 🏆 Gameplay Tips

### For Best Experience

#### Player Positioning
1. **Stand on your side**: Use the center line as a guide
2. **Face the camera**: Both players should face forward
3. **Maintain distance**: Keep 4-6 feet between players
4. **Stay in frame**: Ensure your full body is visible

#### Sword Techniques
1. **Wrist-based detection**: The system tracks your wrist as the sword tip
2. **Clear movements**: Make deliberate, clear striking motions
3. **Avoid rapid movements**: System needs time to process between hits
4. **Target zones**: Aim for opponent's head or torso areas

#### Strategy
1. **Defense**: Move your target zones away from opponent's reach
2. **Timing**: Wait for openings in opponent's defense
3. **Positioning**: Use footwork to create angles
4. **Patience**: Don't rush - accuracy beats speed

### Advanced Tips
1. **Color-coded swords**: Use red/blue tape on sword tips for better tracking
2. **Consistent lighting**: Avoid shadows or bright spots
3. **Warm up**: Practice movements before competitive play
4. **Fair play**: Respect the hit cooldown system

## 🔧 Troubleshooting

### Common Issues

#### "Game not detecting players"
1. **Check lighting**: Ensure even lighting on both sides
2. **Position correctly**: Stand on opposite sides of center line
3. **Full visibility**: Make sure entire body is in frame
4. **Camera height**: Adjust camera to chest height

#### "Only one player detected"
1. **Increase separation**: Move further apart
2. **Check regions**: Ensure each player stays in their half
3. **Lighting balance**: Both sides should be equally lit
4. **Camera distance**: Move camera further back for wider view

#### "Hits not registering"
1. **Clear movements**: Make distinct striking motions
2. **Target the boxes**: Aim for visible bounding boxes
3. **Respect cooldown**: Wait 500ms between hits
4. **Wrist visibility**: Ensure your wrist is visible to camera

#### "Laggy or slow response"
1. **Close other apps**: Free up system resources
2. **Better lighting**: Improve lighting conditions
3. **Camera quality**: Use higher quality camera if available
4. **Reduce complexity**: Lower model complexity in settings

### Performance Issues

#### "Low frame rate"
```python
# In pose_extractor.py, reduce model complexity
extractor = PoseExtractor(model_complexity=0)  # Fastest
```

#### "High CPU usage"
- Close unnecessary applications
- Use lighter pose detection model
- Reduce camera resolution if needed

### Hardware Issues

#### "Camera not found"
1. **Check connections**: Ensure camera is plugged in
2. **Test camera**: Use system camera app to verify
3. **Try different ports**: Use different USB ports
4. **Restart system**: Sometimes helps with driver issues

#### "Poor image quality"
1. **Clean lens**: Clean camera lens
2. **Adjust focus**: If camera has manual focus
3. **Lighting**: Add more even lighting
4. **Camera settings**: Check camera app settings

## 📊 Game Modes

### Training Mode
- Single player practice
- No time limit
- Focus on pose detection accuracy

### Quick Match
- 2 players, first to 3 points
- 30 second time limit
- Standard rules

### Tournament Mode (Future)
- Best of 5 rounds
- Bracket system
- Score tracking

## 🎨 Customization

### Visual Settings
```python
# In game.py, customize colors
PLAYER_1_COLOR = (0, 255, 0)    # Green
PLAYER_2_COLOR = (0, 0, 255)    # Red
HIT_FLASH_COLOR = (0, 255, 255) # Yellow for hits
```

### Game Settings
```python
# In game.py, adjust game parameters
WINNING_SCORE = 3        # Points to win
GAME_DURATION = 30       # Seconds
HIT_COOLDOWN = 0.5       # Seconds between hits
```

### Detection Settings
```python
# In pose_extractor.py, adjust detection
PoseExtractor(
    min_detection_confidence=0.5,  # Lower = more sensitive
    min_tracking_confidence=0.5,   # Lower = more sensitive
    model_complexity=1             # 0=fast, 1=balanced, 2=accurate
)
```

## 🎬 Demo Script

For demonstrations or tournaments, follow this script:

### Setup (5 minutes)
1. Set up camera and test view
2. Ensure good lighting
3. Test pose detection with both players
4. Explain rules and controls

### Demonstration (10 minutes)
1. Show single player detection
2. Demonstrate 2-player detection
3. Show hit detection with bounding boxes
4. Play a quick match

### Tournament Play (Variable)
1. Bracket system with multiple matches
2. Best of 3 or 5 rounds
3. Track scores and announce winners

## 📈 Future Enhancements

### Planned Features
- **Color-based sword detection**: Using colored tape on sword tips
- **Multiple game modes**: Different rule sets and challenges
- **Score persistence**: Save high scores and statistics
- **Tournament bracket**: Built-in tournament management
- **Replay system**: Record and replay matches

### Advanced Features (Possible)
- **3+ player support**: Expand beyond 2 players
- **Different weapons**: Support for different weapon types
- **Special moves**: Combo attacks and special techniques
- **Online multiplayer**: Remote play capabilities

## 📞 Support

### Getting Help
1. **Check this documentation**: Most issues are covered here
2. **Test with pose extractor**: Use `python pose_extractor.py` to debug
3. **Check hardware**: Verify camera and lighting setup
4. **Review logs**: Check console output for error messages

### Common Solutions
- **Restart application**: Often fixes temporary issues
- **Adjust lighting**: Most detection issues are lighting-related
- **Check positioning**: Ensure proper player positioning
- **Update drivers**: Keep camera drivers up to date

---

Have fun playing the sword fighting game! ⚔️🎮
