# Gesture-Controlled Game Collection

This project is a collection of interactive games that use your body as a controller. Using a webcam, your movements are translated into in-game actions, providing a unique and immersive gaming experience.

## Features

-   **Real-time Gesture Recognition:** Utilizes MediaPipe for accurate and real-time pose detection.
-   **Multiple Game Modes:** Includes Sword Fighting, IRL Martial Arts, and Flappy Bird.
-   **Multi-player Support:** The Sword Fighting and IRL Martial Arts games support two players.
-   **3D Stick Figure Rendering:** Visualizes your pose as a 3D stick figure in the game.

## Games

### 1. Sword Fighting

A two-player sword fighting game where each player controls a virtual sword with their body movements. The goal is to hit your opponent to score points.

### 2. IRL Martial Arts Fighting

A gesture-based fighting game where you can perform various attacks like punches, kicks, and blocks by making the corresponding real-life movements.

### 3. Flappy Bird

A classic Flappy Bird game where you control the bird by flapping your arms.

## Dependencies

The project requires the following Python libraries:

-   `mediapipe>=0.10.14`
-   `opencv-python>=4.8.0`
-   `pygame>=2.5.0`
-   `numpy>=1.24.0`
-   `onnxruntime>=1.19.0`
-   `requests>=2.31.0`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To start the game, run the `main.py` script:

```bash
python main.py
```

A menu will appear allowing you to choose which game to play.

## Controls

-   **Navigation:** Use the number keys (1, 2, 3) to select a game from the menu.
-   **In-Game Actions:** Control the games by performing the corresponding physical movements in front of your webcam.
-   **Exit:** Press the `ESC` key to exit the game.