"""
Pose Extractor Module for Sword Fighting Game

This module provides pose detection and tracking functionality using MediaPipe.
It's designed to detect multiple people, extract key landmarks, and provide
utilities for the sword fighting game.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import os
import tensorflow as tf

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Try to import MediaPipe Selfie Segmentation for person detection
try:
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    HAS_SEGMENTATION = True
except:
    HAS_SEGMENTATION = False

def configure_nvidia_gpu():
    """Configure TensorFlow and environment for NVIDIA GPU usage"""
    try:
        # Force NVIDIA GPU usage and bypass version checking
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first NVIDIA GPU
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging
        
        # Try to bypass CUDA version compatibility issues
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        os.environ['TF_CUDA_VERSION'] = '12.8'
        
        # Suppress CUDA version warnings
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        # Configure TensorFlow for GPU with error handling
        try:
            # Try to force TensorFlow to use GPU despite version mismatch
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set GPU as preferred device
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    print(f"✅ Configured TensorFlow to use NVIDIA GPU: {gpus[0]}")
                    return True
                except RuntimeError as e:
                    print(f"⚠️  GPU configuration error: {e}")
                    return False
            else:
                # Try alternative GPU detection methods
                print("⚠️  No NVIDIA GPUs detected by TensorFlow, trying alternative methods...")
                
                # Check if CUDA is available at all
                if tf.test.is_built_with_cuda():
                    print("✅ TensorFlow was built with CUDA support")
                    # Even if GPU isn't detected, MediaPipe might still use GPU
                    return True
                else:
                    print("❌ TensorFlow was not built with CUDA support")
                    return False
                    
        except Exception as gpu_error:
            print(f"⚠️  GPU detection failed: {gpu_error}")
            # Check if we can still use CUDA for MediaPipe
            if tf.test.is_built_with_cuda():
                print("✅ TensorFlow has CUDA support, MediaPipe may still use GPU")
                return True
            return False
            
    except Exception as e:
        print(f"⚠️  Failed to configure NVIDIA GPU: {e}")
        return False

class BodyPart(Enum):
    """Enumeration of body parts for hit detection"""
    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"

@dataclass
class PlayerPose:
    """Data class to store pose information for a player"""
    landmarks: List[Any]
    landmarks_obj: Any  # MediaPipe landmarks object for drawing
    wrist_left: Optional[Tuple[int, int]]
    wrist_right: Optional[Tuple[int, int]]
    body_bounding_box: Optional[Tuple[int, int, int, int]]
    head_bounding_box: Optional[Tuple[int, int, int, int]]
    torso_bounding_box: Optional[Tuple[int, int, int, int]]
    confidence: float
    player_id: int

class PoseExtractor:
    """
    Main pose extraction class that handles pose detection and tracking
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 max_num_poses: int = 2,
                 enable_gpu: bool = True):
        """
        Initialize the pose extractor
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Complexity of the pose model (0, 1, or 2)
            max_num_poses: Maximum number of poses to detect (for multi-person)
            enable_gpu: Whether to try using GPU acceleration
        """
        # For multi-person detection, we'll use multiple pose instances
        # MediaPipe Pose doesn't natively support multi-person, so we'll use a different approach
        self.max_num_poses = max_num_poses
        self.enable_gpu = enable_gpu
        
        # Configure GPU if requested
        self.gpu_configured = False
        if enable_gpu:
            print("Attempting to initialize MediaPipe with NVIDIA GPU support...")
            self.gpu_configured = configure_nvidia_gpu()
        
        # Try to initialize with GPU support
        try:
            if enable_gpu:
                print("Initializing MediaPipe with GPU acceleration...")
                
                # Try to force MediaPipe to use GPU even with TensorFlow issues
                # MediaPipe has its own GPU acceleration separate from TensorFlow
                try:
                    # Set environment for MediaPipe GPU acceleration
                    os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Enable GPU
                    
                    self.pose = mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=model_complexity,
                        enable_segmentation=False,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    
                    if self.gpu_configured:
                        print("🚀 MediaPipe initialized with NVIDIA GPU acceleration!")
                    else:
                        print("🔧 MediaPipe initialized with GPU support (TensorFlow fallback)")
                        
                except Exception as mp_error:
                    print(f"⚠️  MediaPipe GPU initialization failed: {mp_error}")
                    print("Falling back to CPU...")
                    
                    # Disable GPU for MediaPipe and use CPU
                    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                    self.pose = mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=model_complexity,
                        enable_segmentation=False,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    print("✅ MediaPipe initialized with CPU")
            else:
                print("Initializing MediaPipe with CPU only...")
                os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Explicitly disable GPU
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("✅ MediaPipe initialized with CPU")
        except Exception as e:
            print(f"⚠️  MediaPipe initialization failed, using fallback: {e}")
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        
        # Drawing specifications
        self.landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(245, 117, 66), thickness=2, circle_radius=2
        )
        self.connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(245, 66, 230), thickness=2, circle_radius=2
        )
    
    def extract_poses(self, frame: np.ndarray) -> List[PlayerPose]:
        """
        Extract poses from a frame with multi-person support
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of PlayerPose objects
        """
        if self.max_num_poses == 1:
            return self._extract_single_pose(frame)
        else:
            return self._extract_multiple_poses(frame)
    
    def _extract_single_pose(self, frame: np.ndarray) -> List[PlayerPose]:
        """Extract single pose from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.pose.process(rgb_frame)
        
        poses = []
        if results.pose_landmarks:
            pose_data = self._extract_pose_data(
                results.pose_landmarks, 
                frame.shape[1], 
                frame.shape[0],
                player_id=0,
                landmarks_obj=results.pose_landmarks
            )
            if pose_data:
                poses.append(pose_data)
        
        return poses
    
    def _extract_multiple_poses(self, frame: np.ndarray) -> List[PlayerPose]:
        """
        Extract multiple poses using region-based approach
        Since MediaPipe Pose is designed for single person, we'll split the frame
        """
        poses = []
        frame_height, frame_width = frame.shape[:2]
        
        if self.max_num_poses == 2:
            # Split frame into left and right halves for 2-person detection
            mid_x = frame_width // 2
            
            # Left region (Player 0)
            left_region = frame[:, :mid_x]
            left_pose = self._detect_pose_in_region(left_region, 0, 0, 0)
            if left_pose:
                poses.append(left_pose)
            
            # Right region (Player 1)  
            right_region = frame[:, mid_x:]
            right_pose = self._detect_pose_in_region(right_region, mid_x, 0, 1)
            if right_pose:
                poses.append(right_pose)
        else:
            # Fallback to single person detection
            poses = self._extract_single_pose(frame)
        
        return poses
    
    def _detect_pose_in_region(self, region: np.ndarray, offset_x: int, offset_y: int, player_id: int) -> Optional[PlayerPose]:
        """
        Detect pose in a specific region of the frame
        """
        if region.size == 0:
            return None
        
        # Convert to RGB
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        rgb_region.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose.process(rgb_region)
        
        if results.pose_landmarks:
            # Extract pose data with offset to map back to full frame coordinates
            pose_data = self._extract_pose_data_with_offset(
                results.pose_landmarks,
                region.shape[1],  # region width
                region.shape[0],  # region height
                player_id,
                results.pose_landmarks,
                offset_x,
                offset_y
            )
            return pose_data
        
        return None
    
    def _extract_pose_data_with_offset(self, landmarks, region_width: int, region_height: int, 
                                     player_id: int, landmarks_obj=None, offset_x: int = 0, offset_y: int = 0) -> Optional[PlayerPose]:
        """
        Extract pose data with offset for region-based detection
        """
        try:
            # Extract wrist positions with offset
            left_wrist = self._get_landmark_position_with_offset(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST], 
                region_width, region_height, offset_x, offset_y
            )
            right_wrist = self._get_landmark_position_with_offset(
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST], 
                region_width, region_height, offset_x, offset_y
            )
            
            # Extract bounding boxes with offset
            body_box = self._get_body_bounding_box_with_offset(landmarks.landmark, region_width, region_height, offset_x, offset_y)
            head_box = self._get_head_bounding_box_with_offset(landmarks.landmark, region_width, region_height, offset_x, offset_y)
            torso_box = self._get_torso_bounding_box_with_offset(landmarks.landmark, region_width, region_height, offset_x, offset_y)
            
            # Calculate confidence
            confidence = self._calculate_pose_confidence(landmarks.landmark)
            
            return PlayerPose(
                landmarks=landmarks.landmark,
                landmarks_obj=landmarks_obj,
                wrist_left=left_wrist,
                wrist_right=right_wrist,
                body_bounding_box=body_box,
                head_bounding_box=head_box,
                torso_bounding_box=torso_box,
                confidence=confidence,
                player_id=player_id
            )
        except Exception as e:
            print(f"Error extracting pose data with offset: {e}")
            return None
    
    def _get_landmark_position_with_offset(self, landmark, region_width: int, region_height: int, offset_x: int, offset_y: int) -> Optional[Tuple[int, int]]:
        """Convert normalized landmark to pixel coordinates with offset"""
        if landmark.visibility < 0.5:
            return None
        
        x = int(landmark.x * region_width) + offset_x
        y = int(landmark.y * region_height) + offset_y
        return (x, y)
    
    def _get_body_bounding_box_with_offset(self, landmarks, region_width: int, region_height: int, offset_x: int, offset_y: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate body bounding box with offset"""
        try:
            key_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in key_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * region_width + offset_x)
                    y_coords.append(landmark.y * region_height + offset_y)
            
            if len(x_coords) < 3:
                return None
            
            padding = 20
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = int(max(x_coords)) + padding
            y2 = int(max(y_coords)) + padding
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _get_head_bounding_box_with_offset(self, landmarks, region_width: int, region_height: int, offset_x: int, offset_y: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate head bounding box with offset"""
        try:
            head_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_EYE,
                mp_pose.PoseLandmark.RIGHT_EYE,
                mp_pose.PoseLandmark.LEFT_EAR,
                mp_pose.PoseLandmark.RIGHT_EAR
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in head_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * region_width + offset_x)
                    y_coords.append(landmark.y * region_height + offset_y)
            
            if len(x_coords) < 2:
                return None
            
            padding = 30
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = int(max(x_coords)) + padding
            y2 = int(max(y_coords)) + padding
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _get_torso_bounding_box_with_offset(self, landmarks, region_width: int, region_height: int, offset_x: int, offset_y: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate torso bounding box with offset"""
        try:
            torso_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in torso_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * region_width + offset_x)
                    y_coords.append(landmark.y * region_height + offset_y)
            
            if len(x_coords) < 3:
                return None
            
            padding = 15
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = int(max(x_coords)) + padding
            y2 = int(max(y_coords)) + padding
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _extract_pose_data(self, landmarks, frame_width: int, frame_height: int, player_id: int, landmarks_obj=None) -> Optional[PlayerPose]:
        """
        Extract relevant pose data from MediaPipe landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_width: Width of the frame
            frame_height: Height of the frame
            player_id: ID of the player
            landmarks_obj: MediaPipe landmarks object for drawing
            
        Returns:
            PlayerPose object or None if extraction fails
        """
        try:
            # Extract wrist positions (potential sword tips)
            left_wrist = self._get_landmark_position(
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST], 
                frame_width, frame_height
            )
            right_wrist = self._get_landmark_position(
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST], 
                frame_width, frame_height
            )
            
            # Extract bounding boxes for different body parts
            body_box = self._get_body_bounding_box(landmarks.landmark, frame_width, frame_height)
            head_box = self._get_head_bounding_box(landmarks.landmark, frame_width, frame_height)
            torso_box = self._get_torso_bounding_box(landmarks.landmark, frame_width, frame_height)
            
            # Calculate overall pose confidence
            confidence = self._calculate_pose_confidence(landmarks.landmark)
            
            return PlayerPose(
                landmarks=landmarks.landmark,
                landmarks_obj=landmarks_obj,
                wrist_left=left_wrist,
                wrist_right=right_wrist,
                body_bounding_box=body_box,
                head_bounding_box=head_box,
                torso_bounding_box=torso_box,
                confidence=confidence,
                player_id=player_id
            )
        except Exception as e:
            print(f"Error extracting pose data: {e}")
            return None
    
    def _get_landmark_position(self, landmark, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Convert normalized landmark to pixel coordinates"""
        if landmark.visibility < 0.5:  # Only use visible landmarks
            return None
        
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        return (x, y)
    
    def _get_body_bounding_box(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for the entire body"""
        try:
            # Use key landmarks to define the body area
            key_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in key_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * frame_width)
                    y_coords.append(landmark.y * frame_height)
            
            if len(x_coords) < 3:  # Need at least 3 visible landmarks
                return None
            
            # Add some padding to the bounding box
            padding = 20
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = min(frame_width, int(max(x_coords)) + padding)
            y2 = min(frame_height, int(max(y_coords)) + padding)
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _get_head_bounding_box(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for the head"""
        try:
            head_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_EYE,
                mp_pose.PoseLandmark.RIGHT_EYE,
                mp_pose.PoseLandmark.LEFT_EAR,
                mp_pose.PoseLandmark.RIGHT_EAR
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in head_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * frame_width)
                    y_coords.append(landmark.y * frame_height)
            
            if len(x_coords) < 2:
                return None
            
            # Add padding for head area
            padding = 30
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = min(frame_width, int(max(x_coords)) + padding)
            y2 = min(frame_height, int(max(y_coords)) + padding)
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _get_torso_bounding_box(self, landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box for the torso"""
        try:
            torso_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP
            ]
            
            x_coords = []
            y_coords = []
            
            for landmark_id in torso_landmarks:
                landmark = landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    x_coords.append(landmark.x * frame_width)
                    y_coords.append(landmark.y * frame_height)
            
            if len(x_coords) < 3:
                return None
            
            padding = 15
            x1 = max(0, int(min(x_coords)) - padding)
            y1 = max(0, int(min(y_coords)) - padding)
            x2 = min(frame_width, int(max(x_coords)) + padding)
            y2 = min(frame_height, int(max(y_coords)) + padding)
            
            return (x1, y1, x2, y2)
        except:
            return None
    
    def _calculate_pose_confidence(self, landmarks) -> float:
        """Calculate overall confidence of the pose detection"""
        try:
            # Use visibility of key landmarks to calculate confidence
            key_landmarks = [
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            
            total_visibility = sum(landmarks[lm.value].visibility for lm in key_landmarks)
            return total_visibility / len(key_landmarks)
        except:
            return 0.0
    
    def draw_pose(self, frame: np.ndarray, pose: PlayerPose, 
                  landmarks_obj=None,
                  draw_landmarks: bool = True, 
                  draw_bounding_boxes: bool = False,
                  player_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw pose information on the frame
        
        Args:
            frame: Input frame to draw on
            pose: PlayerPose object
            landmarks_obj: MediaPipe landmarks object for drawing
            draw_landmarks: Whether to draw pose landmarks
            draw_bounding_boxes: Whether to draw bounding boxes
            player_color: Color for drawing (BGR format)
            
        Returns:
            Frame with pose drawn
        """
        if draw_landmarks:
            # Instead of using the regional landmarks_obj, draw landmarks manually using corrected coordinates
            self._draw_pose_landmarks_manually(frame, pose, player_color)
            
        if draw_bounding_boxes:
            # Draw body bounding box
            if pose.body_bounding_box:
                cv2.rectangle(frame, 
                            (pose.body_bounding_box[0], pose.body_bounding_box[1]),
                            (pose.body_bounding_box[2], pose.body_bounding_box[3]),
                            player_color, 2)
                cv2.putText(frame, f"Player {pose.player_id}", 
                          (pose.body_bounding_box[0], pose.body_bounding_box[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, player_color, 2)
            
            # Draw head bounding box
            if pose.head_bounding_box:
                cv2.rectangle(frame, 
                            (pose.head_bounding_box[0], pose.head_bounding_box[1]),
                            (pose.head_bounding_box[2], pose.head_bounding_box[3]),
                            (0, 255, 255), 1)  # Yellow for head
            
            # Draw torso bounding box
            if pose.torso_bounding_box:
                cv2.rectangle(frame, 
                            (pose.torso_bounding_box[0], pose.torso_bounding_box[1]),
                            (pose.torso_bounding_box[2], pose.torso_bounding_box[3]),
                            (255, 0, 255), 1)  # Magenta for torso
        
        # Draw wrist positions (potential sword tips)
        if pose.wrist_left:
            cv2.circle(frame, pose.wrist_left, 8, (0, 0, 255), -1)  # Red for left wrist
        if pose.wrist_right:
            cv2.circle(frame, pose.wrist_right, 8, (255, 0, 0), -1)  # Blue for right wrist
        
        return frame
    
    def _draw_pose_landmarks_manually(self, frame: np.ndarray, pose: PlayerPose, color: Tuple[int, int, int]):
        """
        Draw pose landmarks manually using the corrected wrist positions
        This avoids the offset issue from regional detection
        """
        # Draw key landmarks using the corrected positions
        if pose.wrist_left:
            cv2.circle(frame, pose.wrist_left, 5, (0, 0, 255), -1)  # Red for left wrist
        if pose.wrist_right:
            cv2.circle(frame, pose.wrist_right, 5, (255, 0, 0), -1)  # Blue for right wrist
        
        # Draw other key landmarks if available
        frame_height, frame_width = frame.shape[:2]
        
        # Get key landmark positions with corrected coordinates
        key_landmarks_to_draw = [
            (mp_pose.PoseLandmark.NOSE, (255, 255, 0)),  # Yellow nose
            (mp_pose.PoseLandmark.LEFT_SHOULDER, (0, 255, 255)),  # Cyan shoulders
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, (0, 255, 255)),
            (mp_pose.PoseLandmark.LEFT_HIP, (255, 0, 255)),  # Magenta hips
            (mp_pose.PoseLandmark.RIGHT_HIP, (255, 0, 255)),
            (mp_pose.PoseLandmark.LEFT_KNEE, (128, 255, 128)),  # Light green knees
            (mp_pose.PoseLandmark.RIGHT_KNEE, (128, 255, 128)),
        ]
        
        for landmark_id, landmark_color in key_landmarks_to_draw:
            if len(pose.landmarks) > landmark_id.value:
                landmark = pose.landmarks[landmark_id.value]
                if landmark.visibility > 0.5:
                    # Calculate position based on the player's region
                    if pose.player_id == 0:  # Left player
                        x = int(landmark.x * (frame_width // 2))
                        y = int(landmark.y * frame_height)
                    elif pose.player_id == 1:  # Right player
                        x = int(landmark.x * (frame_width // 2)) + (frame_width // 2)
                        y = int(landmark.y * frame_height)
                    else:  # Single player mode
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                    
                    cv2.circle(frame, (x, y), 4, landmark_color, -1)
        
        # Draw connections between key landmarks
        connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST),
        ]
        
        for start_landmark, end_landmark in connections:
            if (len(pose.landmarks) > start_landmark.value and 
                len(pose.landmarks) > end_landmark.value):
                
                start_lm = pose.landmarks[start_landmark.value]
                end_lm = pose.landmarks[end_landmark.value]
                
                if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                    # Calculate positions
                    if pose.player_id == 0:  # Left player
                        start_x = int(start_lm.x * (frame_width // 2))
                        start_y = int(start_lm.y * frame_height)
                        end_x = int(end_lm.x * (frame_width // 2))
                        end_y = int(end_lm.y * frame_height)
                    elif pose.player_id == 1:  # Right player
                        start_x = int(start_lm.x * (frame_width // 2)) + (frame_width // 2)
                        start_y = int(start_lm.y * frame_height)
                        end_x = int(end_lm.x * (frame_width // 2)) + (frame_width // 2)
                        end_y = int(end_lm.y * frame_height)
                    else:  # Single player mode
                        start_x = int(start_lm.x * frame_width)
                        start_y = int(start_lm.y * frame_height)
                        end_x = int(end_lm.x * frame_width)
                        end_y = int(end_lm.y * frame_height)
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
    
    def get_sword_tip_from_wrist(self, pose: PlayerPose, hand: str = "right") -> Optional[Tuple[int, int]]:
        """
        Get sword tip position based on wrist position
        This is a simplified approach - in reality you might want to extend the wrist position
        based on arm angle or use color detection
        
        Args:
            pose: PlayerPose object
            hand: "left" or "right"
            
        Returns:
            Sword tip position or None
        """
        if hand == "left":
            return pose.wrist_left
        elif hand == "right":
            return pose.wrist_right
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()

def run_pose_extraction(camera_index=None, enable_gpu=True):
    """
    Demo function to test the pose extractor
    
    Args:
        camera_index: Camera index to use (None for auto-detection)
        enable_gpu: Whether to use GPU acceleration
    """
    extractor = PoseExtractor(enable_gpu=enable_gpu)
    
    # Auto-detect RGB camera if not specified
    if camera_index is None:
        print("Auto-detecting RGB camera (avoiding IR cameras)...")
        # Prefer index 0 (usually RGB) over index 2 (often IR on Windows)
        for i in [0, 1, 3, 4, 2]:  # Try RGB cameras first, IR camera last
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                # Try to set higher resolution if possible
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    # Basic check for RGB vs IR camera
                    height, width = frame.shape[:2]
                    if width >= 640 and height >= 480:  # Prefer higher resolution cameras
                        camera_index = i
                        print(f"Found RGB camera at index {i} ({width}x{height})")
                        test_cap.release()
                        break
            test_cap.release()
        
        if camera_index is None:
            print("No suitable camera found!")
            return
    
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        print("Please ensure your webcam is connected and not in use by another application.")
        return

    # Try to set higher resolution for full FOV
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Starting pose extraction demo with camera {camera_index}...")
    print(f"Resolution: {actual_width}x{actual_height}")
    print("Controls:")
    print("  - Press 'b' to toggle bounding boxes")
    print("  - Press 'ESC' to exit")
    print("Make sure you're visible to the camera!")
    
    show_bounding_boxes = False
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam. Exiting.")
                break

            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)

            # Extract poses
            poses = extractor.extract_poses(frame)
            
            # Show detection info and split line for 2-person mode
            if extractor.max_num_poses == 2:
                # Draw center line
                mid_x = frame.shape[1] // 2
                cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), (100, 100, 100), 1)
                cv2.putText(frame, "Player 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Player 2", (mid_x + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif extractor.max_num_poses > 1:
                cv2.putText(frame, f"Multi-person mode (max: {extractor.max_num_poses})", 
                           (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw poses
            for i, pose in enumerate(poses):
                color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for Player 1, red for Player 2
                frame = extractor.draw_pose(frame, pose, 
                                          landmarks_obj=pose.landmarks_obj,
                                          draw_landmarks=True,
                                          draw_bounding_boxes=show_bounding_boxes,
                                          player_color=color)
                
                # Display confidence
                if pose.body_bounding_box:
                    cv2.putText(frame, f"Conf: {pose.confidence:.2f}", 
                              (pose.body_bounding_box[0], pose.body_bounding_box[3] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display instructions and pose count
            cv2.putText(frame, f"Poses detected: {len(poses)} | Press 'b' for boxes, 'ESC' to exit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            try:
                cv2.imshow('Pose Extractor Demo', frame)
            except cv2.error as e:
                print(f"Error displaying window: {e}")
                break

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('b'):  # Toggle bounding boxes
                show_bounding_boxes = not show_bounding_boxes
                print(f"Bounding boxes: {'ON' if show_bounding_boxes else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.cleanup()

if __name__ == '__main__':
    run_pose_extraction()