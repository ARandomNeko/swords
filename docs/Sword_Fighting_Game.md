# ⚔️ PVC Sword Fighting Game -- Wildcard Innovation (Enigma Hackathon 2025)

## 🔹 Overview

A **real-time sword fighting game** where players use **PVC pipes as
swords**, and our system detects their movements with a camera.\
The game awards points when a sword **hits the opponent's body**, and
declares a **winner**.

This project is designed for the **Wildcard Innovation track** (Track 4)
of the Enigma Hackathon 2025.

------------------------------------------------------------------------

## 🔹 Tracking Strategy

### Option A: Wrist as Sword Tip (simplest)

-   Use **MediaPipe Pose** to track the wrist joint.\
-   Represent the sword tip as an extension of the wrist.\
-   Detect collisions between wrist (sword tip) and opponent's body
    bounding box.\
    ✅ Very fast, no extra setup.\
    ⚠️ Less accurate if PVC is angled.

### Option B: Color Marker on PVC Tip (preferred)

-   Wrap PVC sword tip with **bright colored tape (red/blue)**.\
-   Use **OpenCV color detection** to find sword tip in camera frames.\
-   Detect collisions: sword tip inside opponent's body bounding box =
    hit.\
    ✅ Clear, accurate detection.\
    ⚠️ Slightly more coding effort.

------------------------------------------------------------------------

## 🔹 Hit Detection Logic

1.  **Define hit zones** using MediaPipe/DensePose:
    -   Head\
    -   Torso
2.  **Track sword tips**:
    -   Wrist (Option A) OR Colored tip (Option B).
3.  **Collision Check**

``` python
def check_hit(sword_tip, opponent_box):
    x, y = sword_tip
    x1, y1, x2, y2 = opponent_box  # opponent bounding box coords
    return x1 <= x <= x2 and y1 <= y <= y2
```

4.  **Scoring**:
    -   If collision detected → +1 point.\
    -   Add 500ms cooldown per hit (avoid multiple hits per swing).

------------------------------------------------------------------------

## 🔹 Gameplay Rules

-   **2 Players (Judges or team members)** fight with PVC swords.\
-   Each hit = +1 point.\
-   Best of 3 points OR 30 seconds.\
-   Live scoreboard displayed on screen.\
-   System announces **Winner** at end.

------------------------------------------------------------------------

## 🔹 Visual Overlay (for Demo Wow Factor)

-   Show **skeleton overlay** with MediaPipe.\
-   Highlight sword tips (red/blue dots).\
-   Opponent's body flashes red when hit.\
-   Live scoreboard:\

```{=html}
<!-- -->
```
    Judge A ⚔️: 2
    Judge B ⚔️: 3 🏆 Winner!

------------------------------------------------------------------------

## 🔹 Hackathon Feasibility (36 Hours)

**Day 1** - Setup MediaPipe + OpenCV.\
- Implement body + wrist detection.\
- Display bounding boxes.

**Day 2** - Add sword tip tracking (color marker).\
- Implement collision detection + scoring.\
- Add scoreboard overlay.\
- Record demo video.

✅ By the end → A fun, interactive **Sword Fighting Game** that judges
can play live!
