"""
Enhanced Fighting Game with Multi-Player Detection and IRL Gesture Controls

Features:
- Multi-player sword fighting with proper detection
- IRL Martial Arts Fighting with gesture-based attacks
- 720p pose extraction with proper scaling
- Complete 3D stick figure rendering
- Real-time gesture recognition for attacks
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Initialize MediaPipe Pose for multi-person detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create multiple pose instances for better multi-person detection
pose_detector_1 = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

pose_detector_2 = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --- Constants and Configuration ---
# Display settings
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Colors for drawing
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (255, 0, 255)
LIME = (0, 255, 0)

PLAYER_1_COLOR = (0, 100, 255)  # Orange-red
PLAYER_2_COLOR = (255, 100, 0)  # Blue

# Enhanced color palette
COLORS = {
    'player1': PLAYER_1_COLOR,
    'player2': PLAYER_2_COLOR,
    'hit_effect': YELLOW,
    'combo_text': CYAN,
    'health_good': GREEN,
    'health_bad': RED,
    'ui_background': (50, 50, 50),
    'ui_text': WHITE,
    'attack_indicator': (0, 255, 255),
    'block_indicator': (255, 255, 0),
}

# Color ranges for sword detection (HSV)
LOWER_RED_1 = np.array([0, 150, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 150, 50])
UPPER_RED_2 = np.array([180, 255, 255])
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([130, 255, 255])

# Enhanced gameplay settings
WINNING_SCORE = 5
GAME_DURATION = 60  # seconds
HIT_COOLDOWN = 0.5  # seconds - half second cooldown between hits
COMBO_WINDOW = 2.0  # seconds
MAX_HEALTH = 100
GESTURE_HOLD_TIME = 0.5  # seconds to hold gesture
ATTACK_VELOCITY_THRESHOLD = 20.0
SWORD_TO_TORSO_RATIO = 1.2  # sword length relative to torso (70% of user height ≈ 120% of torso)
WRIST_COMBINE_DISTANCE = 60  # pixels - distance for wrists to combine into sword
DEPTH_SCALE_FACTOR = 500  # Scale factor for 3D depth calculations

class GameMode(Enum):
    """Game mode selection"""
    MENU = "menu"
    SWORD_FIGHTING = "sword_fighting"
    IRL_FIGHTING = "irl_fighting"

class AttackType(Enum):
    """Types of IRL attacks"""
    PUNCH = "punch"
    KICK = "kick"
    BLOCK = "block"
    UPPERCUT = "uppercut"
    SWEEP = "sweep"

@dataclass
class Player:
    """Enhanced player data structure"""
    id: int
    health: float
    score: int
    combo_count: int
    combo_timer: float
    last_hit_time: float
    last_attack_time: float
    current_attack: Optional[AttackType]
    attack_history: deque
    pose_landmarks: Optional[Any] = None
    body_box: Optional[Tuple[int, int, int, int]] = None
    wrist_positions: List[Tuple[int, int]] = None
    is_blocking: bool = False
    
    def __post_init__(self):
        if self.attack_history is None:
            self.attack_history = deque(maxlen=10)
        if self.wrist_positions is None:
            self.wrist_positions = []

@dataclass
class AttackData:
    """Attack data structure"""
    attack_type: AttackType
    velocity: float
    position: Tuple[int, int]
    timestamp: float
    damage: int

class GestureRecognizer:
    """Real-time gesture recognition for IRL fighting"""
    
    def __init__(self):
        self.gesture_history_size = 10
        self.attack_patterns = self._load_attack_patterns()
    
    def _load_attack_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load attack gesture patterns"""
        return {
            AttackType.PUNCH.value: {
                'description': 'Forward punch - Extend arm forward quickly',
                'damage': 15,
                'requirements': ['arm_extension', 'forward_velocity'],
                'cooldown': 0.5
            },
            AttackType.KICK.value: {
                'description': 'High kick - Raise leg above waist level',
                'damage': 25,
                'requirements': ['leg_raise', 'high_velocity'],
                'cooldown': 0.8
            },
            AttackType.UPPERCUT.value: {
                'description': 'Uppercut - Quick upward arm motion',
                'damage': 30,
                'requirements': ['upward_motion', 'arm_acceleration'],
                'cooldown': 0.7
            },
            AttackType.BLOCK.value: {
                'description': 'Block - Cross arms in front of body',
                'damage': 0,
                'requirements': ['arms_crossed', 'defensive_stance'],
                'cooldown': 0.3
            },
            AttackType.SWEEP.value: {
                'description': 'Leg sweep - Low horizontal leg motion',
                'damage': 20,
                'requirements': ['low_leg_motion', 'horizontal_sweep'],
                'cooldown': 1.0
            }
        }
    
    def detect_attack(self, player: Player, current_time: float) -> Optional[AttackData]:
        """Detect attack gestures from pose landmarks"""
        if not player.pose_landmarks:
            return None

        # Check each attack type
        for attack_type in AttackType:
            if self._check_attack_pattern(player.pose_landmarks, attack_type, current_time):
                # Calculate attack velocity and position
                velocity = self._calculate_gesture_velocity(player, attack_type)
                position = self._get_attack_position(player.pose_landmarks, attack_type)
                damage = self.attack_patterns[attack_type.value]['damage']
                
                return AttackData(attack_type, velocity, position, current_time, damage)
        
        return None
    
    def _check_attack_pattern(self, landmarks, attack_type: AttackType, current_time: float) -> bool:
        """Check if current pose matches attack pattern"""
        try:
            if attack_type == AttackType.PUNCH:
                return self._detect_punch(landmarks)
            elif attack_type == AttackType.KICK:
                return self._detect_kick(landmarks)
            elif attack_type == AttackType.UPPERCUT:
                return self._detect_uppercut(landmarks)
            elif attack_type == AttackType.BLOCK:
                return self._detect_block(landmarks)
            elif attack_type == AttackType.SWEEP:
                return self._detect_sweep(landmarks)
        except (AttributeError, IndexError):
            pass
        
        return False
    
    def _detect_punch(self, landmarks) -> bool:
        """Detect punch gesture - extended arm forward"""
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            # Check if either arm is extended forward (wrist forward of shoulder)
            left_extended = left_wrist.z < left_shoulder.z - 0.1  # Z is depth
            right_extended = right_wrist.z < right_shoulder.z - 0.1
            
            # Check if arm is relatively straight (elbow between shoulder and wrist)
            left_straight = abs(left_elbow.x - (left_shoulder.x + left_wrist.x) / 2) < 0.1
            right_straight = abs(right_elbow.x - (right_shoulder.x + right_wrist.x) / 2) < 0.1
            
            return (left_extended and left_straight) or (right_extended and right_straight)
            
        except (AttributeError, IndexError):
            return False
    
    def _detect_kick(self, landmarks) -> bool:
        """Detect kick gesture - leg raised above waist"""
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Check if either knee is raised significantly above hip level
            left_knee_raised = left_knee.y < left_hip.y - 0.1
            right_knee_raised = right_knee.y < right_hip.y - 0.1
            
            # Check if ankle is also raised (confirming leg lift)
            left_ankle_raised = left_ankle.y < left_hip.y
            right_ankle_raised = right_ankle.y < right_hip.y
            
            return (left_knee_raised and left_ankle_raised) or (right_knee_raised and right_ankle_raised)
            
        except (AttributeError, IndexError):
            return False
    
    def _detect_uppercut(self, landmarks) -> bool:
        """Detect uppercut gesture - upward arm motion"""
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check if wrist is above shoulder (upward motion)
            left_upward = left_wrist.y < left_shoulder.y - 0.1
            right_upward = right_wrist.y < right_shoulder.y - 0.1
            
            return left_upward or right_upward
            
        except (AttributeError, IndexError):
            return False
    
    def _detect_block(self, landmarks) -> bool:
        """Detect block gesture - arms crossed in front"""
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check if wrists are in front of body and close together
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            wrist_center_x = (left_wrist.x + right_wrist.x) / 2
            
            # Arms should be in front of body center
            arms_forward = abs(wrist_center_x - shoulder_center_x) < 0.15
            
            # Wrists should be at chest level (between shoulders and hips)
            chest_level = (left_wrist.y > left_shoulder.y and left_wrist.y < left_shoulder.y + 0.3 and
                          right_wrist.y > right_shoulder.y and right_wrist.y < right_shoulder.y + 0.3)
            
            return arms_forward and chest_level
            
        except (AttributeError, IndexError):
            return False
    
    def _detect_sweep(self, landmarks) -> bool:
        """Detect sweep gesture - low horizontal leg motion"""
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Check if one leg is extended horizontally (sweep motion)
            hip_level = (left_hip.y + right_hip.y) / 2
            
            # One ankle should be at hip level or higher (leg raised horizontally)
            left_horizontal = abs(left_ankle.y - hip_level) < 0.1
            right_horizontal = abs(right_ankle.y - hip_level) < 0.1
            
            return left_horizontal or right_horizontal
            
        except (AttributeError, IndexError):
            return False
    
    def _calculate_gesture_velocity(self, player: Player, attack_type: AttackType) -> float:
        """Calculate velocity of gesture based on movement history"""
        if len(player.attack_history) < 2:
            return 0.0
        
        # Get recent positions for velocity calculation
        recent_attacks = list(player.attack_history)[-3:]
        if len(recent_attacks) < 2:
            return 0.0
        
        # Calculate velocity between recent positions
        velocities = []
        for i in range(1, len(recent_attacks)):
            prev_attack = recent_attacks[i-1]
            curr_attack = recent_attacks[i]
            
            if hasattr(prev_attack, 'position') and hasattr(curr_attack, 'position'):
                dx = curr_attack.position[0] - prev_attack.position[0]
                dy = curr_attack.position[1] - prev_attack.position[1]
                dt = curr_attack.timestamp - prev_attack.timestamp
                
                if dt > 0:
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
                    velocities.append(velocity)
        
        return max(velocities) if velocities else 0.0
    
    def _get_attack_position(self, landmarks, attack_type: AttackType) -> Tuple[int, int]:
        """Get the position of the attack based on type"""
        try:
            if attack_type in [AttackType.PUNCH, AttackType.UPPERCUT]:
                # Use wrist position for hand attacks
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Choose the wrist that's more forward/active
                if left_wrist.z < right_wrist.z:  # Left wrist more forward
                    return (int(left_wrist.x * TARGET_WIDTH), int(left_wrist.y * TARGET_HEIGHT))
                else:
                    return (int(right_wrist.x * TARGET_WIDTH), int(right_wrist.y * TARGET_HEIGHT))
                    
            elif attack_type in [AttackType.KICK, AttackType.SWEEP]:
                # Use ankle position for leg attacks
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                # Choose the ankle that's higher/more active
                if left_ankle.y < right_ankle.y:  # Left ankle higher
                    return (int(left_ankle.x * TARGET_WIDTH), int(left_ankle.y * TARGET_HEIGHT))
                else:
                    return (int(right_ankle.x * TARGET_WIDTH), int(right_ankle.y * TARGET_HEIGHT))
                    
            elif attack_type == AttackType.BLOCK:
                # Use chest center for block
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                center_x = (left_shoulder.x + right_shoulder.x) / 2
                center_y = (left_shoulder.y + right_shoulder.y) / 2
                return (int(center_x * TARGET_WIDTH), int(center_y * TARGET_HEIGHT))
                
        except (AttributeError, IndexError):
            pass
            
        return (TARGET_WIDTH // 2, TARGET_HEIGHT // 2)  # Default center

class IRLFightingGame:
    """IRL Fighting Game with gesture-based attacks"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.gesture_recognizer = GestureRecognizer()
        
        # Initialize players
        self.players = [
            Player(
                id=0, health=MAX_HEALTH, score=0, combo_count=0,
                combo_timer=0, last_hit_time=0, last_attack_time=0,
                current_attack=None, attack_history=deque(maxlen=10)
            ),
            Player(
                id=1, health=MAX_HEALTH, score=0, combo_count=0,
                combo_timer=0, last_hit_time=0, last_attack_time=0,
                current_attack=None, attack_history=deque(maxlen=10)
            )
        ]
        
        self.game_start_time = time.time()
        self.winner = None
        self.hit_effects = []
    
    def update(self, player_poses: List[Any], current_time: float):
        """Update IRL fighting game state"""
        # Update player poses and detect attacks
        for i, player in enumerate(self.players):
            if i < len(player_poses) and player_poses[i]:
                player.pose_landmarks = player_poses[i]
                player.body_box = self._get_body_bounding_box(player_poses[i])
                
                # Detect attacks
                attack = self.gesture_recognizer.detect_attack(player, current_time)
                if attack and current_time - player.last_attack_time > GESTURE_HOLD_TIME:
                    player.current_attack = attack.attack_type
                    player.last_attack_time = current_time
                    player.attack_history.append(attack)
                    
                    # Check for hits on opponent
                    self._check_hit(player, attack, current_time)
                
                # Update blocking state
                if player.pose_landmarks:
                    player.is_blocking = self.gesture_recognizer._detect_block(player.pose_landmarks)
        
        # Update combo timers
        for player in self.players:
            if current_time - player.combo_timer > COMBO_WINDOW:
                player.combo_count = 0
        
        # Update hit effects
        self.hit_effects = [
            effect for effect in self.hit_effects 
            if current_time - effect['time'] < 1.0
        ]
        
        # Check for winner
        self._check_winner(current_time)
    
    def _check_hit(self, attacking_player: Player, attack: AttackData, current_time: float):
        """Check if attack hits opponent"""
        opponent = self.players[1 - attacking_player.id]
        
        if not opponent.body_box or current_time - attacking_player.last_hit_time < HIT_COOLDOWN:
            return
        
        # Check if attack position intersects with opponent's body
        x, y = attack.position
        x1, y1, x2, y2 = opponent.body_box
        
        if x1 <= x <= x2 and y1 <= y <= y2:
            # Check if opponent is blocking
            damage_multiplier = 0.3 if opponent.is_blocking else 1.0
            actual_damage = int(attack.damage * damage_multiplier)
            
            # Apply damage
            opponent.health -= actual_damage
            attacking_player.last_hit_time = current_time
            
            # Update combo
            attacking_player.combo_count += 1
            attacking_player.combo_timer = current_time
            
            # Add hit effect
            self.hit_effects.append({
                'position': attack.position,
                'time': current_time,
                'attack_type': attack.attack_type.value,
                'damage': actual_damage,
                'blocked': opponent.is_blocking
            })
            
            # Update score based on attack type
            score_multiplier = min(attacking_player.combo_count, 5)
            attacking_player.score += score_multiplier
            
            print(f"Player {attacking_player.id + 1} hits with {attack.attack_type.value}! "
                  f"Damage: {actual_damage} {'(BLOCKED)' if opponent.is_blocking else ''}")
    
    def _check_winner(self, current_time: float):
        """Check for game winner"""
        elapsed_time = current_time - self.game_start_time
        
        # Check health
        for player in self.players:
            if player.health <= 0:
                self.winner = f"Player {(1 - player.id) + 1}"
                return
        
        # Check score
        for player in self.players:
            if player.score >= WINNING_SCORE:
                self.winner = f"Player {player.id + 1}"
                return
        
        # Check time
        if elapsed_time >= GAME_DURATION:
            if self.players[0].score > self.players[1].score:
                self.winner = "Player 1"
            elif self.players[1].score > self.players[0].score:
                self.winner = "Player 2"
            else:
                self.winner = "Draw"
    
    def _get_body_bounding_box(self, landmarks) -> Optional[Tuple[int, int, int, int]]:
        """Get body bounding box from pose landmarks"""
        try:
            # Get key body landmarks
            head_landmarks = [
                landmarks[mp_pose.PoseLandmark.NOSE.value],
                      landmarks[mp_pose.PoseLandmark.LEFT_EYE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            ]
                      
            torso_landmarks = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ]

            x_coords = [lm.x * TARGET_WIDTH for lm in head_landmarks + torso_landmarks]
            y_coords = [lm.y * TARGET_HEIGHT for lm in head_landmarks + torso_landmarks]

            if not x_coords or not y_coords:
                return None

            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))

            return (x1, y1, x2, y2)

        except (AttributeError, IndexError):
            return None

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw IRL fighting game UI"""
        # Draw health bars
        for i, player in enumerate(self.players):
            x = 50 if i == 0 else self.width - 250
            y = 50
            
            # Player label
            player_color = COLORS['player1'] if i == 0 else COLORS['player2']
            cv2.putText(frame, f"Player {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, player_color, 2)
            
            # Health bar background
            cv2.rectangle(frame, (x, y), (x + 200, y + 25), COLORS['ui_background'], -1)
            cv2.rectangle(frame, (x, y), (x + 200, y + 25), COLORS['ui_text'], 2)
            
            # Health bar fill
            health_width = int(200 * (player.health / MAX_HEALTH))
            health_color = COLORS['health_good'] if player.health > 50 else COLORS['health_bad']
            cv2.rectangle(frame, (x, y), (x + health_width, y + 25), health_color, -1)
            
            # Health text
            cv2.putText(frame, f"{int(player.health)}/100", (x + 75, y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['ui_text'], 1)
            
            # Score
            cv2.putText(frame, f"Score: {player.score}", (x, y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, player_color, 2)
            
            # Combo counter
            if player.combo_count > 1:
                combo_text = f"{player.combo_count}x COMBO!"
                cv2.putText(frame, combo_text, (x, y + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['combo_text'], 2)
            
            # Current attack indicator
            if player.current_attack:
                attack_text = player.current_attack.value.upper()
                cv2.putText(frame, attack_text, (x, y + 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['attack_indicator'], 2)
            
            # Block indicator
            if player.is_blocking:
                cv2.putText(frame, "BLOCKING", (x, y + 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['block_indicator'], 2)
        
        # Game timer
        elapsed_time = time.time() - self.game_start_time
        time_left = max(0, int(GAME_DURATION - elapsed_time))
        timer_text = f"Time: {time_left}"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        timer_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, timer_text, (timer_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['ui_text'], 2)
        
        # Winner announcement
        if self.winner:
            win_text = f"{self.winner} Wins!" if self.winner != "Draw" else "It's a Draw!"
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = (self.height + text_size[1]) // 2
            
            # Background
            cv2.rectangle(frame, (text_x - 20, text_y - 40), 
                         (text_x + text_size[0] + 20, text_y + 10), COLORS['ui_background'], -1)
            
            # Text
            cv2.putText(frame, win_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS['combo_text'], 3)
        
        # Attack guide
        self.draw_attack_guide(frame)
        
        return frame
    
    def draw_attack_guide(self, frame: np.ndarray):
        """Draw attack gesture guide"""
        guide_y = self.height - 120
        
        # Background
        cv2.rectangle(frame, (10, guide_y), (self.width - 10, guide_y + 110), 
                     COLORS['ui_background'], -1)
        cv2.rectangle(frame, (10, guide_y), (self.width - 10, guide_y + 110), 
                     COLORS['ui_text'], 1)
        
        # Title
        cv2.putText(frame, "ATTACK GESTURES", (20, guide_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['ui_text'], 2)
        
        # Attack descriptions
        attacks = [
            "PUNCH: Extend arm forward quickly",
            "KICK: Raise leg above waist level", 
            "UPPERCUT: Quick upward arm motion",
            "BLOCK: Cross arms in front of body",
            "SWEEP: Low horizontal leg motion"
        ]
        
        for i, attack in enumerate(attacks):
            x = 20 + (i % 3) * 400  # 3 columns
            y = guide_y + 40 + (i // 3) * 20
            cv2.putText(frame, attack, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['ui_text'], 1)
    
    def draw_hit_effects(self, frame: np.ndarray, current_time: float):
        """Draw hit effects"""
        for effect in self.hit_effects:
            age = current_time - effect['time']
            alpha = max(0, 1 - age)
            
            # Hit explosion effect
            radius = int(15 + age * 25)
            color = tuple(int(c * alpha) for c in COLORS['hit_effect'])
            cv2.circle(frame, effect['position'], radius, color, 2)
            
            # Attack type text
            attack_text = effect['attack_type'].upper()
            if effect['blocked']:
                attack_text += " (BLOCKED)"
            
            text_pos = (effect['position'][0] - 40, effect['position'][1] - 30)
            cv2.putText(frame, attack_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, COLORS['combo_text'], 2)
            
            # Damage text
            damage_text = f"-{effect['damage']}"
            damage_pos = (effect['position'][0] - 20, effect['position'][1] - 10)
            cv2.putText(frame, damage_text, damage_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, COLORS['hit_effect'], 2)
    
    def reset_game(self):
        """Reset the fighting game"""
        for player in self.players:
            player.health = MAX_HEALTH
            player.score = 0
            player.combo_count = 0
            player.combo_timer = 0
            player.last_hit_time = 0
            player.last_attack_time = 0
            player.current_attack = None
            player.attack_history.clear()
            player.is_blocking = False
        
        self.game_start_time = time.time()
        self.winner = None
        self.hit_effects.clear()

class EnhancedGameSystem:
    """Main game system with multiple modes and proper multi-player detection"""
    
    def __init__(self):
        self.current_mode = GameMode.MENU
        
        # Initialize components
        self.irl_fighting_game = IRLFightingGame(TARGET_WIDTH, TARGET_HEIGHT)
        
        # Enhanced sword fighting game state
        self.sword_players = [
            Player(
                id=0, health=MAX_HEALTH, score=0, combo_count=0,
                combo_timer=0, last_hit_time=0, last_attack_time=0,
                current_attack=None, attack_history=deque(maxlen=10)
            ),
            Player(
                id=1, health=MAX_HEALTH, score=0, combo_count=0,
                combo_timer=0, last_hit_time=0, last_attack_time=0,
                current_attack=None, attack_history=deque(maxlen=10)
            )
        ]
        
        self.game_start_time = time.time()
        self.winner = None
        
        # Setup camera
        self.setup_camera()
    
    def setup_camera(self):
        """Setup camera with 720p resolution"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video stream")
        
        # Try to set high resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera resolution: {actual_width}x{actual_height}")
        print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
        """Process frame and extract multiple poses"""
        # Resize to target resolution
        if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract poses using region-based multi-person detection
        poses = self._detect_multiple_poses(frame)
        
        return frame, poses
    
    def _detect_multiple_poses(self, frame: np.ndarray) -> List[Any]:
        """Detect multiple poses using frame splitting"""
        poses = []
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB once
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Split frame for two-person detection
        mid_x = frame_width // 2
        
        # Left region (Player 1)
        left_region = rgb_frame[:, :mid_x]
        try:
            left_results = pose_detector_1.process(left_region)
            if left_results.pose_landmarks:
                # Adjust coordinates back to full frame
                adjusted_landmarks = self._adjust_landmarks(
                    left_results.pose_landmarks.landmark, 0, 0, mid_x, frame_height, frame_width, frame_height
                )
                poses.append(adjusted_landmarks)
        except Exception as e:
            print(f"Error processing left region: {e}")
        
        # Right region (Player 2)
        right_region = rgb_frame[:, mid_x:]
        try:
            right_results = pose_detector_2.process(right_region)
            if right_results.pose_landmarks:
                # Adjust coordinates back to full frame
                adjusted_landmarks = self._adjust_landmarks(
                    right_results.pose_landmarks.landmark, mid_x, 0, mid_x, frame_height, frame_width, frame_height
                )
                poses.append(adjusted_landmarks)
        except Exception as e:
            print(f"Error processing right region: {e}")
        
        return poses
    
    def _adjust_landmarks(self, landmarks, offset_x: int, offset_y: int, 
                         region_width: int, region_height: int,
                         full_width: int, full_height: int) -> List[Any]:
        """Adjust landmark coordinates from region to full frame"""
        adjusted = []
        for landmark in landmarks:
            # Create a new landmark-like object with adjusted coordinates
            class AdjustedLandmark:
                def __init__(self, x, y, z, visibility):
                    # Adjust x coordinate from region to full frame
                    self.x = (x * region_width + offset_x) / full_width
                    self.y = (y * region_height + offset_y) / full_height
                    self.z = z
                    self.visibility = visibility
            
            adjusted.append(AdjustedLandmark(landmark.x, landmark.y, landmark.z, landmark.visibility))
        
        return adjusted
    
    def run(self):
        """Main game loop"""
        print("🎮 Enhanced Fighting Game System")
        print("Select mode:")
        print("  [1] Sword Fighting Game (Multi-player)")
        print("  [2] IRL Martial Arts Fighting (Gesture-based)")
        print("  [ESC] Exit")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                current_time = time.time()
                
                # Process frame with multi-player detection
                frame, poses = self.process_frame(frame)
                
                # Handle different modes
                if self.current_mode == GameMode.MENU:
                    frame = self.draw_menu(frame)
                elif self.current_mode == GameMode.SWORD_FIGHTING:
                    frame = self.run_sword_fighting(frame, poses, current_time)
                elif self.current_mode == GameMode.IRL_FIGHTING:
                    frame = self.run_irl_fighting(frame, poses, current_time)
                
                # Display frame
                cv2.imshow('Enhanced Fighting Game System', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif self.current_mode == GameMode.MENU:
                    if key == ord('1'):
                        self.current_mode = GameMode.SWORD_FIGHTING
                        self.reset_sword_fighting()
                    elif key == ord('2'):
                        self.current_mode = GameMode.IRL_FIGHTING
                        self.irl_fighting_game.reset_game()
                elif self.current_mode == GameMode.IRL_FIGHTING:
                    if key == ord('r'):
                        self.irl_fighting_game.reset_game()
                    elif key == ord('m'):
                        self.current_mode = GameMode.MENU
                elif self.current_mode == GameMode.SWORD_FIGHTING:
                    if key == ord('r'):
                        self.reset_sword_fighting()
                    elif key == ord('m'):
                        self.current_mode = GameMode.MENU
        
        except KeyboardInterrupt:
            print("\nExiting...")
        
        finally:
            self.cleanup()
    
    def draw_menu(self, frame: np.ndarray) -> np.ndarray:
        """Draw main menu"""
        # Background
        frame.fill(30)
        
        # Title
        title = "🥊 ENHANCED FIGHTING GAME SYSTEM 🗡️"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (TARGET_WIDTH - text_size[0]) // 2
        cv2.putText(frame, title, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
        
        # Menu options
        options = [
            "1. SWORD FIGHTING GAME - 3D ENHANCED",
            "   • 3D sword with elbow-wrist angle tracking",
            "   • 2D consistent length with 3D visual effects", 
            "   • Advanced arm extension and depth calculations",
            "   • Elbow-weighted sword direction (70% arm, 30% elbow)",
            "   • Enhanced 3D collision detection with 0.5s cooldown",
            "",
            "2. IRL MARTIAL ARTS FIGHTING",
            "   • Gesture-based fighting with real attacks",
            "   • Punch, Kick, Uppercut, Block, Sweep moves",
            "   • Health system with combo multipliers",
            "   • Real-time gesture recognition",
            "   • Blocking system reduces damage",
            "",
            "Press [1] for 3D Sword Fighting or [2] for IRL Fighting",
            "Press [ESC] to exit"
        ]
        
        y_start = 150
        for i, option in enumerate(options):
            if option.startswith(('1.', '2.')):
                color = COLORS['player1'] if option.startswith('1.') else COLORS['player2']
            elif option.startswith('Press'):
                color = YELLOW
            else:
                color = WHITE
            
            cv2.putText(frame, option, (50, y_start + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Instructions
        instructions = [
            "🎮 CONTROLS:",
            "Sword Fighting: Use red/blue colored objects as swords",
            "IRL Fighting: Use your body gestures to attack and defend",
            "Both modes: [R] Restart game, [M] Return to menu"
        ]
        
        inst_y = y_start + len(options) * 30 + 50
        for i, inst in enumerate(instructions):
            color = CYAN if inst.startswith('🎮') else WHITE
            cv2.putText(frame, inst, (50, inst_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_sword_fighting(self, frame: np.ndarray, poses: List[Any], current_time: float) -> np.ndarray:
        """Run enhanced sword fighting game mode with multi-player detection"""
        frame_height, frame_width = frame.shape[:2]
        
        # Convert for HSV processing
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Draw center line for player regions
        mid_x = frame_width // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (100, 100, 100), 1)
        cv2.putText(frame, "Player 1", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['player1'], 2)
        cv2.putText(frame, "Player 2", (mid_x + 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['player2'], 2)
        
        # Draw poses and swords
        body_boxes = []
        sword_active = [False, False]  # Track which players have active swords
        
        for i, pose_landmarks in enumerate(poses[:2]):  # Max 2 players
            if pose_landmarks:
                player_color = COLORS['player1'] if i == 0 else COLORS['player2']
                
                # Draw stick figure
                self._draw_enhanced_stick_figure(frame, pose_landmarks, player_color)
                
                # Draw realistic sword when wrists combine
                has_sword = self._draw_realistic_sword(frame, pose_landmarks, player_color)
                sword_active[i] = has_sword
                
                if has_sword:
                    # Draw sword status indicator
                    status_text = f"Player {i+1}: SWORD ACTIVE"
                    cv2.putText(frame, status_text, (50 + i * 400, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, player_color, 2)
                
                # Get bounding box
                box = self._get_body_bounding_box_from_landmarks(pose_landmarks, frame_width, frame_height)
                body_boxes.append(box)
                
                # Update player data
                if i < len(self.sword_players):
                    self.sword_players[i].pose_landmarks = pose_landmarks
                    self.sword_players[i].body_box = box
        
        # Get sword tips for active swords
        sword_tip_p1 = None
        sword_tip_p2 = None
        
        # Try color detection first
        color_tip_p1 = self.find_sword_tip(frame, hsv_frame, LOWER_RED_1, UPPER_RED_1, LOWER_RED_2, UPPER_RED_2)
        color_tip_p2 = self.find_sword_tip(frame, hsv_frame, LOWER_BLUE, UPPER_BLUE)
        
        # Get realistic sword tips for each player
        for i, pose_landmarks in enumerate(poses[:2]):
            if pose_landmarks and sword_active[i]:
                realistic_tip = self._get_realistic_sword_tip(pose_landmarks, frame_width, frame_height)
                
                if i == 0:
                    sword_tip_p1 = color_tip_p1 if color_tip_p1 else realistic_tip
                elif i == 1:
                    sword_tip_p2 = color_tip_p2 if color_tip_p2 else realistic_tip
        
        # Draw sword tip indicators
        if sword_tip_p1:
            cv2.circle(frame, sword_tip_p1, 10, COLORS['player1'], -1)
            cv2.circle(frame, sword_tip_p1, 12, WHITE, 2)
            cv2.putText(frame, "TIP", (sword_tip_p1[0] - 15, sword_tip_p1[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
        if sword_tip_p2:
            cv2.circle(frame, sword_tip_p2, 10, COLORS['player2'], -1)
            cv2.circle(frame, sword_tip_p2, 12, WHITE, 2)
            cv2.putText(frame, "TIP", (sword_tip_p2[0] - 15, sword_tip_p2[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
        # Advanced hit detection with realistic sword collision
        if not self.winner and len(poses) >= 2:
            # Get sword and body segments for collision detection
            player1_sword = None
            player2_sword = None
            player1_body = []
            player2_body = []
            
            if poses[0] and sword_active[0]:  # Player 1
                player1_sword = self._get_sword_line_segment(poses[0], frame_width, frame_height)
                player1_body = self._get_body_line_segments(poses[0], frame_width, frame_height)
            
            if poses[1] and sword_active[1]:  # Player 2
                player2_sword = self._get_sword_line_segment(poses[1], frame_width, frame_height)
                player2_body = self._get_body_line_segments(poses[1], frame_width, frame_height)
            
            # Player 1 hits Player 2 (only if Player 1 has active sword)
            if player1_sword and player2_body:
                if self._check_sword_body_collision([player1_sword], player2_body):
                    if current_time - self.sword_players[0].last_hit_time > HIT_COOLDOWN:
                        self.sword_players[0].score += 1
                        self.sword_players[0].last_hit_time = current_time
                        print("Player 1 hits Player 2 with sword!")
                        
                        # Enhanced visual hit effect
                        for body_start, body_end in player2_body:
                            cv2.line(frame, body_start, body_end, COLORS['hit_effect'], 8)
                        
                        # Draw hit indicator
                        hit_center = (
                            (player1_sword[0][0] + player1_sword[1][0]) // 2,
                            (player1_sword[0][1] + player1_sword[1][1]) // 2
                        )
                        cv2.putText(frame, "HIT!", (hit_center[0] - 20, hit_center[1] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['hit_effect'], 3)
            
            # Player 2 hits Player 1 (only if Player 2 has active sword)
            if player2_sword and player1_body:
                if self._check_sword_body_collision([player2_sword], player1_body):
                    if current_time - self.sword_players[1].last_hit_time > HIT_COOLDOWN:
                        self.sword_players[1].score += 1
                        self.sword_players[1].last_hit_time = current_time
                        print("Player 2 hits Player 1 with sword!")
                        
                        # Enhanced visual hit effect
                        for body_start, body_end in player1_body:
                            cv2.line(frame, body_start, body_end, COLORS['hit_effect'], 8)
                        
                        # Draw hit indicator
                        hit_center = (
                            (player2_sword[0][0] + player2_sword[1][0]) // 2,
                            (player2_sword[0][1] + player2_sword[1][1]) // 2
                        )
                        cv2.putText(frame, "HIT!", (hit_center[0] - 20, hit_center[1] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['hit_effect'], 3)
            
            # Fallback to point-based detection for sword tips
            if len(body_boxes) > 1 and sword_tip_p1 and body_boxes[1]:
                if self.check_hit(sword_tip_p1, body_boxes[1]):
                    if current_time - self.sword_players[0].last_hit_time > HIT_COOLDOWN:
                        self.sword_players[0].score += 1
                        self.sword_players[0].last_hit_time = current_time
                        print("Player 1 hits Player 2! (fallback)")
            
            if len(body_boxes) > 0 and sword_tip_p2 and body_boxes[0]:
                if self.check_hit(sword_tip_p2, body_boxes[0]):
                    if current_time - self.sword_players[1].last_hit_time > HIT_COOLDOWN:
                        self.sword_players[1].score += 1
                        self.sword_players[1].last_hit_time = current_time
                        print("Player 2 hits Player 1! (fallback)")
        
        # Check winner
        elapsed_time = current_time - self.game_start_time
        if not self.winner:
            if self.sword_players[0].score >= WINNING_SCORE:
                self.winner = "Player 1"
            elif self.sword_players[1].score >= WINNING_SCORE:
                self.winner = "Player 2"
        elif elapsed_time >= GAME_DURATION:
            if self.sword_players[0].score > self.sword_players[1].score:
                self.winner = "Player 1"
            elif self.sword_players[1].score > self.sword_players[0].score:
                self.winner = "Player 2"
            else:
                self.winner = "Draw"
        
        # Draw UI
        cv2.putText(frame, f"Player 1: {self.sword_players[0].score}", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['player1'], 2)
        cv2.putText(frame, f"Player 2: {self.sword_players[1].score}", (frame_width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['player2'], 2)
        
        time_left = max(0, int(GAME_DURATION - elapsed_time))
        cv2.putText(frame, f"Time: {time_left}", (frame_width // 2 - 50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
        
        if self.winner:
            win_text = f"{self.winner} Wins!" if self.winner != "Draw" else "It's a Draw!"
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.putText(frame, win_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS['combo_text'], 3)
        
        # Controls
        cv2.putText(frame, "[R] Restart  [M] Menu", (20, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        return frame
    
    def run_irl_fighting(self, frame: np.ndarray, poses: List[Any], current_time: float) -> np.ndarray:
        """Run IRL martial arts fighting mode"""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw center line for player regions
        mid_x = frame_width // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (100, 100, 100), 1)
        
        # Draw poses with enhanced stick figures
        for i, pose_landmarks in enumerate(poses[:2]):
            if pose_landmarks:
                color = COLORS['player1'] if i == 0 else COLORS['player2']
                self._draw_enhanced_stick_figure(frame, pose_landmarks, color)
        
        # Update IRL fighting game
        self.irl_fighting_game.update(poses, current_time)
        
        # Draw game UI
        frame = self.irl_fighting_game.draw_ui(frame)
        
        # Draw hit effects
        self.irl_fighting_game.draw_hit_effects(frame, current_time)
        
        # Controls
        cv2.putText(frame, "[R] Restart  [M] Menu", (20, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        return frame
    
    def _draw_enhanced_stick_figure(self, frame: np.ndarray, landmarks: List[Any], color: Tuple[int, int, int], draw_swords: bool = False):
        """Draw enhanced 3D stick figure with optional sword objects"""
        h, w = frame.shape[:2]
        
        # Define body connections with thickness
        connections = [
            # Torso (thicker lines)
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 4),
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, 3),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, 3),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value, 4),
            
            # Arms
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, 3),
            (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, 2),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, 3),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, 2),
            
            # Legs
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, 3),
            (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, 2),
            (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, 3),
            (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, 2),
            
            # Head/neck
            (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, 2),
            (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 2),
        ]
        
        for start_idx, end_idx, thickness in connections:
            try:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                    
                    # Add shadow effect
                    shadow_color = tuple(int(c * 0.3) for c in color)
                    cv2.line(frame, 
                           (start_pos[0] + 2, start_pos[1] + 2),
                           (end_pos[0] + 2, end_pos[1] + 2),
                           shadow_color, thickness)
                    
                    # Main line
                    cv2.line(frame, start_pos, end_pos, color, thickness)
                    
            except (AttributeError, IndexError):
                continue
        
        # Draw virtual swords if enabled
        if draw_swords:
            self._draw_virtual_swords(frame, landmarks, color)
        
        # Draw joints
        joint_landmarks = [
            mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, 
            mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, 
            mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.LEFT_HIP.value, 
            mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, 
            mp_pose.PoseLandmark.RIGHT_KNEE.value
        ]
        
        for joint_idx in joint_landmarks:
            try:
                joint_lm = landmarks[joint_idx]
                
                if joint_lm.visibility > 0.5:
                    pos = (int(joint_lm.x * w), int(joint_lm.y * h))
                    radius = 4
                    
                    # Shadow
                    cv2.circle(frame, (pos[0] + 1, pos[1] + 1), radius,
                             tuple(int(c * 0.3) for c in color), -1)
                    
                    # Main joint
                    cv2.circle(frame, pos, radius, color, -1)
                    
                    # Highlight
                    cv2.circle(frame, pos, radius - 1, 
                             tuple(min(255, int(c * 1.5)) for c in color), 1)
                    
            except (AttributeError, IndexError):
                continue
    
    def _draw_realistic_sword(self, frame: np.ndarray, landmarks: List[Any], player_color: Tuple[int, int, int]) -> bool:
        """Draw single realistic 3D sword when wrists combine, considering wrist angle"""
        h, w = frame.shape[:2]
        
        try:
            # Get wrist and elbow positions for 3D angle calculation
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            if (left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5 or
                left_elbow.visibility < 0.5 or right_elbow.visibility < 0.5):
                return False
            
            # Calculate 3D positions
            left_wrist_3d = (left_wrist.x * w, left_wrist.y * h, left_wrist.z * DEPTH_SCALE_FACTOR)
            right_wrist_3d = (right_wrist.x * w, right_wrist.y * h, right_wrist.z * DEPTH_SCALE_FACTOR)
            left_elbow_3d = (left_elbow.x * w, left_elbow.y * h, left_elbow.z * DEPTH_SCALE_FACTOR)
            right_elbow_3d = (right_elbow.x * w, right_elbow.y * h, right_elbow.z * DEPTH_SCALE_FACTOR)
            
            # Calculate 3D distance between wrists
            wrist_distance_3d = math.sqrt(
                (left_wrist_3d[0] - right_wrist_3d[0])**2 + 
                (left_wrist_3d[1] - right_wrist_3d[1])**2 +
                (left_wrist_3d[2] - right_wrist_3d[2])**2
            )
            
            # Only draw sword if wrists are close enough (combined)
            if wrist_distance_3d > WRIST_COMBINE_DISTANCE:
                return False
            
            # Calculate torso height for sword scaling
            torso_height = self._get_torso_height(landmarks, w, h)
            if torso_height == 0:
                return False
            
            # Scale sword based on torso height - 2D length calculation for consistency
            sword_length = int(torso_height * SWORD_TO_TORSO_RATIO)
            
            # Keep sword length consistent in 2D but use 3D logic for direction and effects
            sword_width = max(6, int(sword_length * 0.04))
            handle_length = int(sword_length * 0.25)
            
            # Calculate 3D sword position and direction
            sword_base_3d = (
                (left_wrist_3d[0] + right_wrist_3d[0]) / 2,
                (left_wrist_3d[1] + right_wrist_3d[1]) / 2,
                (left_wrist_3d[2] + right_wrist_3d[2]) / 2
            )
            
            # Calculate enhanced 3D sword direction considering both elbows and wrists
            # Get individual arm directions from elbows to wrists
            left_arm_dir = (
                left_wrist_3d[0] - left_elbow_3d[0],
                left_wrist_3d[1] - left_elbow_3d[1],
                left_wrist_3d[2] - left_elbow_3d[2]
            )
            
            right_arm_dir = (
                right_wrist_3d[0] - right_elbow_3d[0],
                right_wrist_3d[1] - right_elbow_3d[1],
                right_wrist_3d[2] - right_elbow_3d[2]
            )
            
            # Calculate elbow center for additional context
            elbow_center_3d = (
                (left_elbow_3d[0] + right_elbow_3d[0]) / 2,
                (left_elbow_3d[1] + right_elbow_3d[1]) / 2,
                (left_elbow_3d[2] + right_elbow_3d[2]) / 2
            )
            
            # Calculate combined wrist-to-elbow direction for sword orientation
            wrist_to_elbow_dir = (
                sword_base_3d[0] - elbow_center_3d[0],
                sword_base_3d[1] - elbow_center_3d[1],
                sword_base_3d[2] - elbow_center_3d[2]
            )
            
            # Weight the arm directions: 70% individual arm directions, 30% wrist-to-elbow
            avg_arm_dir = (
                (left_arm_dir[0] + right_arm_dir[0]) * 0.7 + wrist_to_elbow_dir[0] * 0.3,
                (left_arm_dir[1] + right_arm_dir[1]) * 0.7 + wrist_to_elbow_dir[1] * 0.3,
                (left_arm_dir[2] + right_arm_dir[2]) * 0.7 + wrist_to_elbow_dir[2] * 0.3
            )
            
            # Normalize 3D direction vector
            arm_length_3d = math.sqrt(avg_arm_dir[0]**2 + avg_arm_dir[1]**2 + avg_arm_dir[2]**2)
            if arm_length_3d > 0:
                unit_x = avg_arm_dir[0] / arm_length_3d
                unit_y = avg_arm_dir[1] / arm_length_3d
                unit_z = avg_arm_dir[2] / arm_length_3d
            else:
                unit_x, unit_y, unit_z = 0, -1, 0  # Default upward
            
            # Project 3D sword to 2D screen coordinates
            sword_base = (int(sword_base_3d[0]), int(sword_base_3d[1]))
            
            # Calculate 3D sword positions
            blade_end_3d = (
                sword_base_3d[0] + unit_x * sword_length,
                sword_base_3d[1] + unit_y * sword_length,
                sword_base_3d[2] + unit_z * sword_length
            )
            
            handle_end_3d = (
                sword_base_3d[0] - unit_x * handle_length,
                sword_base_3d[1] - unit_y * handle_length,
                sword_base_3d[2] - unit_z * handle_length
            )
            
            # Project to 2D with depth perspective
            blade_start = sword_base
            blade_end = (
                int(blade_end_3d[0]),
                int(blade_end_3d[1])
            )
            
            handle_start = sword_base
            handle_end = (
                int(handle_end_3d[0]),
                int(handle_end_3d[1])
            )
            
            # Apply 3D depth effects considering both wrists and elbows
            # Calculate average depth for visual effects
            avg_wrist_depth = (left_wrist_3d[2] + right_wrist_3d[2]) / 2
            avg_elbow_depth = (left_elbow_3d[2] + right_elbow_3d[2]) / 2
            combined_depth = (avg_wrist_depth + avg_elbow_depth) / 2  # Average arm depth
            
            # Calculate arm extension factor (how extended the arms are)
            left_arm_length = math.sqrt(left_arm_dir[0]**2 + left_arm_dir[1]**2 + left_arm_dir[2]**2)
            right_arm_length = math.sqrt(right_arm_dir[0]**2 + right_arm_dir[1]**2 + right_arm_dir[2]**2)
            avg_arm_extension = (left_arm_length + right_arm_length) / 2
            extension_factor = min(1.5, max(0.8, avg_arm_extension / 100))  # Scale factor based on arm extension
            
            depth_factor = 1.0 + (unit_z * 0.3) * extension_factor  # Visual thickness when pointing forward
            depth_brightness_factor = 1.0 + (combined_depth / DEPTH_SCALE_FACTOR) * 0.2  # Depth-based brightness
            
            effective_sword_width = max(4, int(sword_width * depth_factor))
            
            # Enhanced 3D sword colors with combined depth shading
            combined_brightness = depth_brightness_factor * (1.0 + (unit_z * 0.3))  # Combined depth effects
            sword_blade_color = tuple(min(255, int(c * 1.8 * combined_brightness)) for c in player_color)
            sword_handle_color = tuple(min(255, int(c * 0.9 * combined_brightness)) for c in player_color)
            sword_edge_color = tuple(min(255, int(c * combined_brightness)) for c in WHITE)
            
            # Calculate 3D shadow with proper depth offset
            shadow_offset_x = max(2, int(sword_length * 0.015)) + int(unit_z * 3)
            shadow_offset_y = max(2, int(sword_length * 0.015)) + int(abs(unit_z) * 2)
            shadow_color = tuple(int(c * 0.15) for c in player_color)
            
            # Shadow blade with 3D depth
            cv2.line(frame, 
                   (blade_start[0] + shadow_offset_x, blade_start[1] + shadow_offset_y),
                   (blade_end[0] + shadow_offset_x, blade_end[1] + shadow_offset_y),
                   shadow_color, effective_sword_width + 2)
            
            # Shadow handle
            cv2.line(frame,
                   (handle_start[0] + shadow_offset_x, handle_start[1] + shadow_offset_y),
                   (handle_end[0] + shadow_offset_x, handle_end[1] + shadow_offset_y),
                   shadow_color, effective_sword_width)
            
            # Draw main sword blade with 3D gradient effect
            cv2.line(frame, blade_start, blade_end, sword_blade_color, effective_sword_width)
            
            # Draw blade edges for enhanced 3D effect
            edge_width = max(2, effective_sword_width // 3)
            cv2.line(frame, blade_start, blade_end, sword_edge_color, edge_width)
            
            # Draw fuller (blood groove) with 3D perspective
            fuller_offset = sword_length * 0.1
            fuller_start = (
                int(blade_start[0] + unit_x * fuller_offset),
                int(blade_start[1] + unit_y * fuller_offset)
            )
            fuller_end = (
                int(blade_end[0] - unit_x * fuller_offset),
                int(blade_end[1] - unit_y * fuller_offset)
            )
            fuller_width = max(1, effective_sword_width // 4)
            cv2.line(frame, fuller_start, fuller_end, 
                   tuple(int(c * 0.7) for c in sword_blade_color), fuller_width)
            
            # Draw sword handle with 3D depth
            cv2.line(frame, handle_start, handle_end, sword_handle_color, effective_sword_width)
            
            # Draw 3D crossguard (perpendicular to sword direction) - 2D length with 3D visual effects
            crossguard_length = int(sword_length * 0.3)  # Keep 2D proportional length
            crossguard_visual_length = int(crossguard_length * depth_factor)  # Apply 3D visual scaling
            
            # Calculate 3D perpendicular vectors
            # Cross product of sword direction with up vector for crossguard
            up_vector = (0, 0, 1)
            cross_x = unit_y * up_vector[2] - unit_z * up_vector[1]
            cross_y = unit_z * up_vector[0] - unit_x * up_vector[2]
            
            # Normalize crossguard direction
            cross_length = math.sqrt(cross_x**2 + cross_y**2)
            if cross_length > 0:
                cross_x /= cross_length
                cross_y /= cross_length
            else:
                cross_x, cross_y = -unit_y, unit_x  # Fallback 2D perpendicular
            
            crossguard_start = (
                int(sword_base[0] - cross_x * crossguard_visual_length//2),
                int(sword_base[1] - cross_y * crossguard_visual_length//2)
            )
            crossguard_end = (
                int(sword_base[0] + cross_x * crossguard_visual_length//2),
                int(sword_base[1] + cross_y * crossguard_visual_length//2)
            )
            
            crossguard_width = effective_sword_width + 2
            cv2.line(frame, crossguard_start, crossguard_end, sword_handle_color, crossguard_width)
            cv2.line(frame, crossguard_start, crossguard_end, sword_edge_color, max(2, crossguard_width // 2))
            
            # Draw pommel with 2D base size and 3D visual scaling
            base_pommel_radius = max(5, int(sword_length * 0.03))  # 2D base size
            pommel_radius = int(base_pommel_radius * depth_factor)  # 3D visual scaling
            pommel_pos = handle_end
            cv2.circle(frame, pommel_pos, pommel_radius, sword_handle_color, -1)
            cv2.circle(frame, pommel_pos, pommel_radius, sword_edge_color, 2)
            cv2.circle(frame, pommel_pos, max(2, pommel_radius//2), sword_edge_color, -1)
            
            # Draw blade tip with 2D base size and 3D visual enhancement
            base_tip_radius = max(3, int(sword_length * 0.025))  # 2D base size
            tip_radius = int(base_tip_radius * depth_factor)  # 3D visual scaling
            tip_pos = blade_end
            cv2.circle(frame, tip_pos, tip_radius, sword_blade_color, -1)
            cv2.circle(frame, tip_pos, tip_radius, sword_edge_color, 2)
            cv2.circle(frame, tip_pos, max(1, tip_radius//2), WHITE, -1)  # Sharp point
            
            # Enhanced 3D sparkle effects with 2D base calculations
            import random
            if random.random() < 0.6:  # Higher chance for 3D sword
                base_sparkle_count = max(2, int(sword_length * 0.02))  # 2D base count
                sparkle_count = int(base_sparkle_count * depth_brightness_factor)  # 3D visual scaling
                for _ in range(sparkle_count):
                    # Sparkles along the blade with 3D distribution
                    blade_progress = random.uniform(0.2, 0.9)  # Along blade length
                    sparkle_x = int(blade_start[0] + (blade_end[0] - blade_start[0]) * blade_progress)
                    sparkle_y = int(blade_start[1] + (blade_end[1] - blade_start[1]) * blade_progress)
                    
                    # Add some random offset with 3D depth variation
                    offset_range = int(8 * depth_factor)
                    sparkle_x += random.randint(-offset_range, offset_range)
                    sparkle_y += random.randint(-offset_range, offset_range)
                    
                    sparkle_pos = (sparkle_x, sparkle_y)
                    base_sparkle_size = max(1, int(sword_length * 0.01))  # 2D base size
                    sparkle_size = random.randint(1, max(2, int(base_sparkle_size * depth_factor)))  # 3D visual size
                    cv2.circle(frame, sparkle_pos, sparkle_size, WHITE, -1)
            
            return True  # Sword was drawn
            
        except (AttributeError, IndexError, ZeroDivisionError):
            return False
    
    def _get_torso_height(self, landmarks: List[Any], frame_width: int, frame_height: int) -> float:
        """Calculate torso height for sword scaling"""
        try:
            # Get shoulder and hip landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
                left_hip.visibility < 0.5 or right_hip.visibility < 0.5):
                return 0
            
            # Calculate shoulder center and hip center
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) * frame_height / 2
            hip_center_y = (left_hip.y + right_hip.y) * frame_height / 2
            
            # Return torso height
            return abs(shoulder_center_y - hip_center_y)
            
        except (AttributeError, IndexError):
            return 0
    
    def _get_realistic_sword_tip(self, landmarks: List[Any], frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Get the 3D tip position of realistic sword when wrists combine, considering elbows"""
        try:
            # Get wrist and elbow positions for 3D calculation
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            if (left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5 or
                left_elbow.visibility < 0.5 or right_elbow.visibility < 0.5):
                return None
            
            # Calculate 3D positions
            left_wrist_3d = (left_wrist.x * frame_width, left_wrist.y * frame_height, left_wrist.z * DEPTH_SCALE_FACTOR)
            right_wrist_3d = (right_wrist.x * frame_width, right_wrist.y * frame_height, right_wrist.z * DEPTH_SCALE_FACTOR)
            left_elbow_3d = (left_elbow.x * frame_width, left_elbow.y * frame_height, left_elbow.z * DEPTH_SCALE_FACTOR)
            right_elbow_3d = (right_elbow.x * frame_width, right_elbow.y * frame_height, right_elbow.z * DEPTH_SCALE_FACTOR)
            
            # Calculate 3D distance between wrists
            wrist_distance_3d = math.sqrt(
                (left_wrist_3d[0] - right_wrist_3d[0])**2 + 
                (left_wrist_3d[1] - right_wrist_3d[1])**2 +
                (left_wrist_3d[2] - right_wrist_3d[2])**2
            )
            
            # Only return tip if wrists are close enough (combined)
            if wrist_distance_3d > WRIST_COMBINE_DISTANCE:
                return None
            
            # Calculate torso height for sword scaling
            torso_height = self._get_torso_height(landmarks, frame_width, frame_height)
            if torso_height == 0:
                return None
            
            # Scale sword based on torso height - use 2D length for consistency
            sword_length = int(torso_height * SWORD_TO_TORSO_RATIO)
            
            # Calculate 3D sword position and direction
            sword_base_3d = (
                (left_wrist_3d[0] + right_wrist_3d[0]) / 2,
                (left_wrist_3d[1] + right_wrist_3d[1]) / 2,
                (left_wrist_3d[2] + right_wrist_3d[2]) / 2
            )
            
            # Calculate 3D arm direction
            left_arm_dir = (
                left_wrist_3d[0] - left_elbow_3d[0],
                left_wrist_3d[1] - left_elbow_3d[1],
                left_wrist_3d[2] - left_elbow_3d[2]
            )
            
            right_arm_dir = (
                right_wrist_3d[0] - right_elbow_3d[0],
                right_wrist_3d[1] - right_elbow_3d[1],
                right_wrist_3d[2] - right_elbow_3d[2]
            )
            
            # Average arm direction
            avg_arm_dir = (
                (left_arm_dir[0] + right_arm_dir[0]) / 2,
                (left_arm_dir[1] + right_arm_dir[1]) / 2,
                (left_arm_dir[2] + right_arm_dir[2]) / 2
            )
            
            # Normalize 3D direction
            arm_length_3d = math.sqrt(avg_arm_dir[0]**2 + avg_arm_dir[1]**2 + avg_arm_dir[2]**2)
            if arm_length_3d > 0:
                unit_x = avg_arm_dir[0] / arm_length_3d
                unit_y = avg_arm_dir[1] / arm_length_3d
                unit_z = avg_arm_dir[2] / arm_length_3d
            else:
                unit_x, unit_y, unit_z = 0, -1, 0
            
            # Calculate 3D sword tip position
            tip_3d = (
                sword_base_3d[0] + unit_x * sword_length,
                sword_base_3d[1] + unit_y * sword_length,
                sword_base_3d[2] + unit_z * sword_length
            )
            
            # Project to 2D
            tip_pos = (int(tip_3d[0]), int(tip_3d[1]))
            
            return tip_pos
            
        except (AttributeError, IndexError, ZeroDivisionError):
            return None
    
    def _get_sword_line_segment(self, landmarks: List[Any], frame_width: int, frame_height: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get the 3D realistic sword line segment for precise collision detection, considering elbows"""
        try:
            # Get wrist and elbow positions for 3D calculation
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            if (left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5 or
                left_elbow.visibility < 0.5 or right_elbow.visibility < 0.5):
                return None
            
            # Calculate 3D positions
            left_wrist_3d = (left_wrist.x * frame_width, left_wrist.y * frame_height, left_wrist.z * DEPTH_SCALE_FACTOR)
            right_wrist_3d = (right_wrist.x * frame_width, right_wrist.y * frame_height, right_wrist.z * DEPTH_SCALE_FACTOR)
            left_elbow_3d = (left_elbow.x * frame_width, left_elbow.y * frame_height, left_elbow.z * DEPTH_SCALE_FACTOR)
            right_elbow_3d = (right_elbow.x * frame_width, right_elbow.y * frame_height, right_elbow.z * DEPTH_SCALE_FACTOR)
            
            # Calculate 3D distance between wrists
            wrist_distance_3d = math.sqrt(
                (left_wrist_3d[0] - right_wrist_3d[0])**2 + 
                (left_wrist_3d[1] - right_wrist_3d[1])**2 +
                (left_wrist_3d[2] - right_wrist_3d[2])**2
            )
            
            # Only return segment if wrists are close enough (combined)
            if wrist_distance_3d > WRIST_COMBINE_DISTANCE:
                return None
            
            # Calculate torso height for sword scaling
            torso_height = self._get_torso_height(landmarks, frame_width, frame_height)
            if torso_height == 0:
                return None
            
            # Scale sword based on torso height - use 2D length for consistency
            sword_length = int(torso_height * SWORD_TO_TORSO_RATIO)
            
            # Calculate 3D sword position and direction
            sword_base_3d = (
                (left_wrist_3d[0] + right_wrist_3d[0]) / 2,
                (left_wrist_3d[1] + right_wrist_3d[1]) / 2,
                (left_wrist_3d[2] + right_wrist_3d[2]) / 2
            )
            
            # Calculate 3D arm direction
            left_arm_dir = (
                left_wrist_3d[0] - left_elbow_3d[0],
                left_wrist_3d[1] - left_elbow_3d[1],
                left_wrist_3d[2] - left_elbow_3d[2]
            )
            
            right_arm_dir = (
                right_wrist_3d[0] - right_elbow_3d[0],
                right_wrist_3d[1] - right_elbow_3d[1],
                right_wrist_3d[2] - right_elbow_3d[2]
            )
            
            # Average arm direction
            avg_arm_dir = (
                (left_arm_dir[0] + right_arm_dir[0]) / 2,
                (left_arm_dir[1] + right_arm_dir[1]) / 2,
                (left_arm_dir[2] + right_arm_dir[2]) / 2
            )
            
            # Normalize 3D direction
            arm_length_3d = math.sqrt(avg_arm_dir[0]**2 + avg_arm_dir[1]**2 + avg_arm_dir[2]**2)
            if arm_length_3d > 0:
                unit_x = avg_arm_dir[0] / arm_length_3d
                unit_y = avg_arm_dir[1] / arm_length_3d
                unit_z = avg_arm_dir[2] / arm_length_3d
            else:
                unit_x, unit_y, unit_z = 0, -1, 0
            
            # Calculate 3D sword blade line segment
            blade_start_3d = sword_base_3d
            blade_end_3d = (
                sword_base_3d[0] + unit_x * sword_length,
                sword_base_3d[1] + unit_y * sword_length,
                sword_base_3d[2] + unit_z * sword_length
            )
            
            # Project to 2D for collision detection
            blade_start = (int(blade_start_3d[0]), int(blade_start_3d[1]))
            blade_end = (int(blade_end_3d[0]), int(blade_end_3d[1]))
            
            return (blade_start, blade_end)
            
        except (AttributeError, IndexError, ZeroDivisionError):
            return None
    
    def _get_body_line_segments(self, landmarks: List[Any], frame_width: int, frame_height: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get body line segments for precise sword collision detection"""
        segments = []
        
        # Define body connections for collision detection
        body_connections = [
            # Torso (main target areas)
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
            
            # Arms
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
            
            # Legs
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
            (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            
            # Head/neck
            (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            (mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        ]
        
        for start_idx, end_idx in body_connections:
            try:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    start_pos = (int(start_lm.x * frame_width), int(start_lm.y * frame_height))
                    end_pos = (int(end_lm.x * frame_width), int(end_lm.y * frame_height))
                    segments.append((start_pos, end_pos))
                    
            except (AttributeError, IndexError):
                continue
        
        return segments
    
    def _line_intersect(self, line1_start: Tuple[int, int], line1_end: Tuple[int, int], 
                       line2_start: Tuple[int, int], line2_end: Tuple[int, int]) -> bool:
        """Check if two line segments intersect using cross product method"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A, B = line1_start, line1_end
        C, D = line2_start, line2_end
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def _point_to_line_distance(self, point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
        """Calculate minimum distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate line length
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Calculate distance
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / (line_length**2)))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        return math.sqrt((x0 - projection_x)**2 + (y0 - projection_y)**2)
    
    def _check_sword_body_collision(self, sword_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                                   body_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                   collision_threshold: float = 20.0) -> bool:
        """Check if sword segments collide with body segments"""
        for sword_start, sword_end in sword_segments:
            for body_start, body_end in body_segments:
                # Check direct line intersection
                if self._line_intersect(sword_start, sword_end, body_start, body_end):
                    return True
                
                # Check if sword tip is close to body segment
                tip_distance = self._point_to_line_distance(sword_end, body_start, body_end)
                if tip_distance < collision_threshold:
                    return True
                
                # Check if sword blade passes close to body joints
                joint_distance_start = self._point_to_line_distance(body_start, sword_start, sword_end)
                joint_distance_end = self._point_to_line_distance(body_end, sword_start, sword_end)
                
                if joint_distance_start < collision_threshold or joint_distance_end < collision_threshold:
                    return True
        
        return False
    
    def _get_body_bounding_box_from_landmarks(self, landmarks: List[Any], frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Get body bounding box from pose landmarks"""
        try:
            # Get key body landmarks
            head_landmarks = [
                landmarks[mp_pose.PoseLandmark.NOSE.value],
                landmarks[mp_pose.PoseLandmark.LEFT_EYE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            ]
            
            torso_landmarks = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ]
            
            x_coords = [lm.x * frame_width for lm in head_landmarks + torso_landmarks]
            y_coords = [lm.y * frame_height for lm in head_landmarks + torso_landmarks]
            
            if not x_coords or not y_coords:
                return None
            
            x1 = int(min(x_coords))
            y1 = int(min(y_coords))
            x2 = int(max(x_coords))
            y2 = int(max(y_coords))
            
            return (x1, y1, x2, y2)
            
        except (AttributeError, IndexError):
            return None
    
    # Sword fighting helper functions
    def find_sword_tip(self, frame, hsv_frame, lower_color1, upper_color1, lower_color2=None, upper_color2=None):
        """Detects the largest contour for a given color range and returns its center."""
        mask1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
        mask = mask1
        if lower_color2 is not None and upper_color2 is not None:
            mask2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)
            mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None

    def check_hit(self, sword_tip, opponent_box):
        """Checks if the sword tip is inside the opponent's bounding box."""
        if sword_tip is None or opponent_box is None:
            return False
        x, y = sword_tip
        x1, y1, x2, y2 = opponent_box
        return x1 <= x <= x2 and y1 <= y <= y2

    def reset_sword_fighting(self):
        """Reset sword fighting game"""
        for player in self.sword_players:
            player.score = 0
            player.health = MAX_HEALTH
            player.combo_count = 0
            player.combo_timer = 0
            player.last_hit_time = 0
            player.last_attack_time = 0
            player.current_attack = None
            player.attack_history.clear()
            player.is_blocking = False
        
        self.game_start_time = time.time()
        self.winner = None
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        # Clean up pose detectors
        if hasattr(pose_detector_1, 'close'):
            pose_detector_1.close()
        if hasattr(pose_detector_2, 'close'):
            pose_detector_2.close()

if __name__ == "__main__":
    try:
        print("🎮 Starting Enhanced Fighting Game System...")
        print("✅ Multi-player pose detection enabled")
        print("✅ IRL gesture-based fighting enabled")
        print("✅ Enhanced 3D stick figure rendering enabled")
        print()
        
        game_system = EnhancedGameSystem()
        game_system.run()
    except Exception as e:
        print(f"❌ Error starting game system: {e}")
        import traceback
        traceback.print_exc()