# 🎯 Pose Extractor Guide

A comprehensive guide to using the pose extractor for the sword fighting game.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🔹 Overview

The pose extractor uses MediaPipe to detect human poses in real-time from camera input. It's designed specifically for the sword fighting game but can be used for any pose detection application.

### Key Features
- **Single and Multi-Person Detection**: Supports 1-2 people simultaneously
- **RGB Camera Auto-Detection**: Automatically finds and uses RGB cameras (avoids IR cameras)
- **High Resolution Support**: Attempts to use up to 1920x1080 resolution
- **Real-Time Processing**: Optimized for live video processing
- **Accurate Landmark Detection**: Precise wrist and body part tracking
- **Interactive Controls**: Live preview with toggleable features

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam (RGB camera recommended)
- Linux/Windows/macOS

### Dependencies
The pose extractor requires these packages (automatically installed in the Nix environment):
```
mediapipe
opencv-python
numpy
```

### Using Nix (Recommended)
```bash
# The environment is automatically set up with flake.nix
direnv allow  # If using direnv
# or
nix develop
```

## ⚡ Quick Start

### Basic Usage
```bash
# Run the interactive pose extractor demo
python pose_extractor.py
```

### Controls
- **'b'**: Toggle bounding boxes on/off
- **'ESC'**: Exit the application

## 📖 Usage

### 1. Single Person Detection
```python
from pose_extractor import PoseExtractor

# Initialize for single person
extractor = PoseExtractor(max_num_poses=1)

# Process frame
poses = extractor.extract_poses(frame)

# Get pose data
if poses:
    pose = poses[0]
    wrist_position = pose.wrist_right
    body_box = pose.body_bounding_box
```

### 2. Two Person Detection
```python
from pose_extractor import PoseExtractor

# Initialize for two people
extractor = PoseExtractor(max_num_poses=2)

# Process frame
poses = extractor.extract_poses(frame)

# Handle multiple players
for pose in poses:
    if pose.player_id == 0:  # Player 1 (left side)
        player1_wrist = pose.wrist_right
        player1_target = pose.torso_bounding_box
    elif pose.player_id == 1:  # Player 2 (right side)
        player2_wrist = pose.wrist_right
        player2_target = pose.torso_bounding_box
```

### 3. Camera Configuration
```python
# Specify camera index manually
extractor = PoseExtractor()
run_pose_extraction(camera_index=0)  # Use specific camera

# Auto-detection (default)
run_pose_extraction()  # Automatically finds RGB camera
```

## 📚 API Reference

### Classes

#### `PoseExtractor`
Main class for pose detection.

**Constructor:**
```python
PoseExtractor(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    max_num_poses: int = 2
)
```

**Parameters:**
- `min_detection_confidence`: Minimum confidence for pose detection (0.0-1.0)
- `min_tracking_confidence`: Minimum confidence for pose tracking (0.0-1.0)  
- `model_complexity`: Model complexity (0=light, 1=full, 2=heavy)
- `max_num_poses`: Maximum number of people to detect (1 or 2)

**Methods:**

##### `extract_poses(frame: np.ndarray) -> List[PlayerPose]`
Extract poses from a video frame.

**Parameters:**
- `frame`: Input frame in BGR format

**Returns:**
- List of `PlayerPose` objects

##### `draw_pose(frame, pose, landmarks_obj=None, draw_landmarks=True, draw_bounding_boxes=False, player_color=(0,255,0))`
Draw pose visualization on frame.

##### `cleanup()`
Clean up resources. Call when done.

#### `PlayerPose`
Data class containing pose information.

**Attributes:**
- `landmarks`: Raw MediaPipe landmarks
- `landmarks_obj`: MediaPipe landmarks object for drawing
- `wrist_left`: Left wrist position (x, y) or None
- `wrist_right`: Right wrist position (x, y) or None
- `body_bounding_box`: Full body bounding box (x1, y1, x2, y2) or None
- `head_bounding_box`: Head bounding box (x1, y1, x2, y2) or None
- `torso_bounding_box`: Torso bounding box (x1, y1, x2, y2) or None
- `confidence`: Pose detection confidence (0.0-1.0)
- `player_id`: Player identifier (0 or 1)

### Functions

#### `run_pose_extraction(camera_index=None)`
Run interactive pose extraction demo.

**Parameters:**
- `camera_index`: Camera index to use (None for auto-detection)

## 🎮 For Game Development

### Sword Tip Tracking
```python
# Get sword tip position (using wrist as proxy)
sword_tip = pose.wrist_right  # or pose.wrist_left

# Check if sword tip is valid
if sword_tip:
    x, y = sword_tip
    # Use sword tip coordinates for game logic
```

### Hit Detection Zones
```python
# Different target zones for hit detection
head_zone = pose.head_bounding_box      # Head hits
torso_zone = pose.torso_bounding_box    # Body hits  
full_body = pose.body_bounding_box      # Any hit

# Check if sword hits target
def check_hit(sword_tip, target_zone):
    if sword_tip and target_zone:
        x, y = sword_tip
        x1, y1, x2, y2 = target_zone
        return x1 <= x <= x2 and y1 <= y <= y2
    return False
```

### Multi-Player Setup
```python
# For 2-player sword fighting
extractor = PoseExtractor(max_num_poses=2)
poses = extractor.extract_poses(frame)

# Players are assigned based on position:
# Player 0: Left side of screen
# Player 1: Right side of screen

for pose in poses:
    player_num = pose.player_id + 1  # 1-indexed for display
    print(f"Player {player_num} detected with confidence {pose.confidence:.2f}")
```

## 🔧 Troubleshooting

### Camera Issues

#### "No camera found" or "Could not open video stream"
1. **Check camera connection**: Ensure webcam is plugged in
2. **Close other applications**: Make sure no other app is using the camera
3. **Try different camera index**: 
   ```bash
   # Test different camera indices
   python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(5)]"
   ```
4. **Check permissions**: Ensure camera permissions are granted

#### "Using IR camera instead of RGB"
- The system prioritizes RGB cameras, but if only IR camera is available, it will use it
- Check if you have multiple cameras and specify the RGB camera manually

### Performance Issues

#### "Slow processing" or "Low FPS"
1. **Reduce model complexity**: Use `model_complexity=0` for faster processing
2. **Lower resolution**: The system will adapt to camera's maximum resolution
3. **Reduce confidence thresholds**: Lower `min_detection_confidence` and `min_tracking_confidence`

#### "High CPU usage"
1. **Use lighter model**: Set `model_complexity=0`
2. **Reduce frame rate**: Add delays in your processing loop
3. **Process every nth frame**: Skip frames for better performance

### Detection Issues

#### "Not detecting poses" or "Low confidence"
1. **Improve lighting**: Ensure good lighting conditions
2. **Check visibility**: Make sure people are fully visible in frame
3. **Adjust confidence**: Lower `min_detection_confidence` to 0.3
4. **Check distance**: Stand at appropriate distance from camera

#### "Only detecting one person in 2-person mode"
1. **Position correctly**: Stand on opposite sides of the center line
2. **Maintain distance**: Keep some space between people
3. **Check visibility**: Both people should be fully visible in their respective halves
4. **Good lighting**: Ensure both sides are well-lit

### Display Issues

#### "Qt platform plugin errors"
- These are warnings and don't affect functionality
- The application will still work correctly

#### "Window not showing"
- Check if running in headless environment
- Ensure display is available
- Try running with `DISPLAY` environment variable set

## 💡 Tips for Best Results

### Camera Setup
- **Use RGB camera**: Avoid IR/infrared cameras when possible
- **Good lighting**: Ensure even lighting across the scene
- **Stable mounting**: Mount camera at chest height, 6-10 feet away
- **Wide angle**: Use camera with wide field of view for 2-person detection

### For 2-Person Detection
- **Stand on opposite sides**: Use the center line as a guide
- **Maintain separation**: Keep at least 3 feet between people
- **Face the camera**: Both people should face toward the camera
- **Avoid overlap**: Don't cross into the other person's region

### Performance Optimization
- **Close other applications**: Free up system resources
- **Use wired camera**: USB cameras often perform better than wireless
- **Adequate hardware**: Ensure sufficient CPU/GPU for real-time processing

## 📝 Example Integration

Here's a complete example of integrating the pose extractor into a game:

```python
import cv2
from pose_extractor import PoseExtractor

class SwordFightingGame:
    def __init__(self):
        self.extractor = PoseExtractor(max_num_poses=2)
        self.cap = cv2.VideoCapture(0)
        
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            
            # Extract poses
            poses = self.extractor.extract_poses(frame)
            
            # Game logic
            self.update_game_state(poses)
            
            # Visualization
            self.draw_game_overlay(frame, poses)
            
            cv2.imshow('Sword Fighting Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
    
    def update_game_state(self, poses):
        # Implement your game logic here
        for pose in poses:
            sword_tip = pose.wrist_right
            target_zone = pose.torso_bounding_box
            # ... hit detection logic
    
    def draw_game_overlay(self, frame, poses):
        # Draw game-specific overlays
        for pose in poses:
            color = (0, 255, 0) if pose.player_id == 0 else (0, 0, 255)
            frame = self.extractor.draw_pose(frame, pose, 
                                           draw_bounding_boxes=True,
                                           player_color=color)
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.cleanup()

if __name__ == "__main__":
    game = SwordFightingGame()
    game.run()
```

---

For more information, see the other documentation files in the `docs/` folder.
