# ⚔️ PVC Sword Fighting Game -- Project Report

## 🔹 Team Information

-   **Team Name:** \[Your Team Name\]\
-   **Members:** \[List of members with roles -- Research, Development,
    Robotics/Hardware\]

------------------------------------------------------------------------

## 🔹 Problem Statement (Wildcard Innovation -- Track 4)

Most hackathon projects fall into productivity, sustainability, or
accessibility domains.\
We wanted to create something **fun, interactive, and visually
striking** that showcases real-time computer vision.

**Our idea:**\
A **camera-based sword fighting game** where players use **PVC pipes as
swords**.\
The system detects their movements, awards points when hits land, and
declares a winner.

------------------------------------------------------------------------

## 🔹 Solution Overview

-   **Input:** Camera feed of two players fighting with PVC swords.\
-   **Processing:**
    -   Body & joint detection using **MediaPipe Pose** (fallback:
        DensePose).\
    -   Sword tip detection via **wrist tracking** or **colored marker
        on PVC tip** (OpenCV).\
    -   Collision detection between sword tip and opponent's body zones
        (head/torso).\
    -   Score updates + cooldown to avoid multiple hits per swing.\
-   **Output:**
    -   Live camera feed with overlays (skeleton, sword tips, flashes on
        hit).\
    -   Real-time scoreboard.\
    -   Winner announcement.

------------------------------------------------------------------------

## 🔹 Technical Approach

### Components

1.  **Computer Vision (Development):**
    -   MediaPipe for real-time skeleton detection.\
    -   OpenCV for color detection of PVC sword tips.\
    -   Collision detection logic in Python.
2.  **Game Logic (Development):**
    -   Hit zones (head, torso).\
    -   Point system (best of 3 / time-based).\
    -   Live scoreboard overlay.
3.  **Testing & Hardware (Robotics/Hardware):**
    -   PVC pipes with colored tape as swords.\
    -   Camera setup in hackathon room.\
    -   Testing hit accuracy + gameplay rules.
4.  **Research (Research role):**
    -   Explored DensePose vs MediaPipe trade-offs.\
    -   Chose MediaPipe for speed + hackathon feasibility.\
    -   Studied simple collision models for hit detection.

------------------------------------------------------------------------

## 🔹 Challenges Faced

-   Balancing **accuracy vs real-time speed** (DensePose too heavy,
    fallback to MediaPipe).\
-   Detecting PVC pipes reliably → solved with **color marker**.\
-   Preventing duplicate hits → solved with **cooldown timer**.

------------------------------------------------------------------------

## 🔹 Future Improvements

-   Add **multiple hit zones** (arms, legs).\
-   Use **3D pose estimation** for more realistic sword tracking.\
-   Turn into a **full AR sword-fighting game** with virtual swords +
    effects.\
-   Multiplayer networked version (remote sword fights).

------------------------------------------------------------------------

## 🔹 Demo Instructions

1.  Stand two players in front of camera with PVC swords.\
2.  Start the program → live feed with overlays appears.\
3.  Players swing swords → hits flash + points scored.\
4.  System announces winner after 30s or 3 points.

------------------------------------------------------------------------

## 🔹 Conclusion

Our project, **PVC Sword Fighting Game**, reimagines hackathon
creativity by turning real-life motion into a fun, interactive computer
vision experience.\
It demonstrates **real-time body tracking, game logic, and visual
overlays** -- showing both technical execution and entertainment value.

------------------------------------------------------------------------

## 🔹 Deliverables

-   **Code:** \[GitHub Repo Link\]\
-   **Demo Video:** \[YouTube/Drive Link\]\
-   **Report:** This document (PDF/Markdown)
