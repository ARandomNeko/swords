import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Constants and Configuration ---
# Colors for drawing
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
PLAYER_1_COLOR = RED
PLAYER_2_COLOR = BLUE

# Color ranges for sword detection (HSV)
# These ranges will likely need to be tuned based on lighting and the specific tape used.
# Player 1 (Red)
LOWER_RED_1 = np.array([0, 150, 50])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 150, 50])
UPPER_RED_2 = np.array([180, 255, 255])
# Player 2 (Blue)
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([130, 255, 255])

# Gameplay settings
WINNING_SCORE = 3
GAME_DURATION = 30  # in seconds
HIT_COOLDOWN = 0.5  # in seconds

# --- Game State ---
player_1_score = 0
player_2_score = 0
last_hit_time_p1 = 0
last_hit_time_p2 = 0
game_start_time = time.time()
winner = None

# --- Helper Functions ---

def find_sword_tip(frame, hsv_frame, lower_color1, upper_color1, lower_color2=None, upper_color2=None):
    """Detects the largest contour for a given color range and returns its center."""
    mask1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
    mask = mask1
    if lower_color2 is not None and upper_color2 is not None:
        mask2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)
        mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Minimum contour area to be considered a sword
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

def get_body_bounding_box(landmarks, frame_width, frame_height):
    """Calculates the bounding box for the torso and head."""
    # Using specific landmarks for head and torso
    # Head: nose, left_eye, right_eye
    # Torso: left_shoulder, right_shoulder, left_hip, right_hip
    
    head_landmarks = [landmarks[mp_pose.PoseLandmark.NOSE.value],
                      landmarks[mp_pose.PoseLandmark.LEFT_EYE.value],
                      landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]]
                      
    torso_landmarks = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]]

    x_coords = [lm.x * frame_width for lm in head_landmarks + torso_landmarks]
    y_coords = [lm.y * frame_height for lm in head_landmarks + torso_landmarks]

    if not x_coords or not y_coords:
        return None

    x1 = int(min(x_coords))
    y1 = int(min(y_coords))
    x2 = int(max(x_coords))
    y2 = int(max(y_coords))

    return (x1, y1, x2, y2)


def check_hit(sword_tip, opponent_box):
    """Checks if the sword tip is inside the opponent's bounding box."""
    if sword_tip is None or opponent_box is None:
        return False
    x, y = sword_tip
    x1, y1, x2, y2 = opponent_box
    return x1 <= x <= x2 and y1 <= y <= y2

# --- Main Game Loop ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the BGR image to RGB and HSV
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # --- Player and Sword Detection ---
    player_landmarks = []
    if results.pose_landmarks:
        # For simplicity, we'll assume the person on the left is Player 1
        # and the person on the right is Player 2.
        # A more robust solution would involve tracking individuals.
        
        # This is a simplified way to distinguish two people.
        # It finds all detected poses and splits them based on their x-coordinates.
        all_landmarks = results.pose_multi_landmarks
        if len(all_landmarks) >= 1:
            # Sort poses by the x-coordinate of the nose
            all_landmarks = sorted(all_landmarks, key=lambda lm: lm.landmark[mp_pose.PoseLandmark.NOSE.value].x)
            
            # Draw poses and get bounding boxes
            body_boxes = []
            for i, landmarks in enumerate(all_landmarks):
                mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
                player_landmarks.append(landmarks.landmark)
                box = get_body_bounding_box(landmarks.landmark, frame_width, frame_height)
                body_boxes.append(box)
                
                # Assign player number based on position
                player_num = i + 1
                if box:
                    cv2.putText(frame, f"Player {player_num}", (box[0], box[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)


    # Find sword tips
    sword_tip_p1 = find_sword_tip(frame, hsv_frame, LOWER_RED_1, UPPER_RED_1, LOWER_RED_2, UPPER_RED_2)
    sword_tip_p2 = find_sword_tip(frame, hsv_frame, LOWER_BLUE, UPPER_BLUE)

    if sword_tip_p1:
        cv2.circle(frame, sword_tip_p1, 10, PLAYER_1_COLOR, -1)
    if sword_tip_p2:
        cv2.circle(frame, sword_tip_p2, 10, PLAYER_2_COLOR, -1)

    # --- Hit Detection and Scoring ---
    current_time = time.time()
    
    if not winner:
        # Player 1 hits Player 2
        if len(body_boxes) > 1 and check_hit(sword_tip_p1, body_boxes[1]):
            if current_time - last_hit_time_p1 > HIT_COOLDOWN:
                player_1_score += 1
                last_hit_time_p1 = current_time
                print("Player 1 scores!")
                if body_boxes[1]:
                    cv2.rectangle(frame, (body_boxes[1][0], body_boxes[1][1]), (body_boxes[1][2], body_boxes[1][3]), RED, 3)


        # Player 2 hits Player 1
        if len(body_boxes) > 0 and check_hit(sword_tip_p2, body_boxes[0]):
            if current_time - last_hit_time_p2 > HIT_COOLDOWN:
                player_2_score += 1
                last_hit_time_p2 = current_time
                print("Player 2 scores!")
                if body_boxes[0]:
                    cv2.rectangle(frame, (body_boxes[0][0], body_boxes[0][1]), (body_boxes[0][2], body_boxes[0][3]), RED, 3)


    # --- Check for Winner ---
    elapsed_time = current_time - game_start_time
    if not winner:
        if player_1_score >= WINNING_SCORE:
            winner = "Player 1"
        elif player_2_score >= WINNING_SCORE:
            winner = "Player 2"
        elif elapsed_time >= GAME_DURATION:
            if player_1_score > player_2_score:
                winner = "Player 1"
            elif player_2_score > player_1_score:
                winner = "Player 2"
            else:
                winner = "Draw"

    # --- Display Information ---
    # Scoreboard
    cv2.putText(frame, f"Player 1: {player_1_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, PLAYER_1_COLOR, 2)
    cv2.putText(frame, f"Player 2: {player_2_score}", (frame_width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, PLAYER_2_COLOR, 2)

    # Timer
    time_left = max(0, int(GAME_DURATION - elapsed_time))
    cv2.putText(frame, f"Time: {time_left}", (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    # Winner announcement
    if winner:
        win_text = f"{winner} Wins!"
        if winner == "Draw":
            win_text = "It's a Draw!"
        text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        cv2.putText(frame, win_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, GREEN, 3)


    # --- Show Frame ---
    cv2.imshow('Sword Fighting Game', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()