# AERO-POSE

**A**gnostic **E**rgonomic **R**eal-time **O**bservation for **P**ose-**O**riented **S**afety **E**valuation

A camera-agnostic, multi-person monitoring system that delivers medical-grade ergonomic feedback regardless of camera placement.

**Core Innovation:** By transforming 2D pixel coordinates into 3D spatial vectors, AERO-Pose removes viewpoint dependency and reconstructs the human body in a virtual 3D space for consistent accuracy.

---

## How It Works

```
Camera → YOLOv8-Pose → 2D Keypoints → VideoPose3D → 3D Joints → REBA Score → Overlay
                  (17 COCO joints)    (243-frame TCN)  (17 H36M joints)  (1-15 scale)
```

1. **Detection** — YOLOv8-Pose detects people and extracts 17 2D keypoints per frame
2. **Lifting** — A temporal convolutional network (VideoPose3D) lifts 2D keypoints to 3D joint positions using a 243-frame sliding window
3. **Ergonomics** — Joint angles are computed from 3D bone vectors and mapped through REBA (Rapid Entire Body Assessment) scoring tables
4. **Feedback** — Real-time overlay shows the 2D skeleton, REBA score (color-coded by risk level), and FPS

---

## Quickstart

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- Camera or video file

### Create Environment

```bash
conda create -n aero-pose python=3.11 -y
conda activate aero-pose
```

### Install

```bash
pip install -e .
```

### Download Model Weights

```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin \
  -O models/videopose3d_243.bin
```

### Run

```bash
# With webcam (default)
aero-pose run

# With video file
aero-pose run --source /path/to/video.mp4

# With specific camera
aero-pose run --source 1

# With larger YOLO model for better accuracy
aero-pose run --model s
```

Press `q` or `Esc` to quit.

### Interactive Controls (OpenCV UI)
The OpenCV UI now features an interactive dashboard layout with the camera feed on the left and detailed ergonomic information on the right.

- **Mouse Drag (Left Click)**: Click and drag on the 3D viewport to rotate the 3D human model (adjusting azimuth and elevation).
- **`q` / `Esc`**: Quit the application.

### Web Dashboard
The system now supports a real-time Web Dashboard interface using FastAPI and Three.js, offering a modern and interactive experience.

```bash
# Start the FastAPI server
python -m aero_pose.api.server
```

Access the dashboard at `http://localhost:8000` in your web browser. It features smooth 3D rotation, synchronized 2D/3D overlays, and detailed REBA scores.

---

## System Architecture

```
aero_pose/
├── main.py                   # CLI entry point (Typer)
├── config/settings.py        # Pydantic configuration model
├── camera/
│   └── capture.py            # Camera/webcam/file capture
├── detection/
│   └── yolo_pose.py          # YOLOv8-Pose detector
├── tracking/
│   └── tracker.py            # Person identity tracking
├── lifting/
│   ├── model.py              # VideoPose3D temporal convolution model
│   ├── single_view.py        # 2D→3D lifting with sliding window
│   └── spatial_vectors.py    # COCO→H36M joint mapping, 3D math
├── reconstruction/
│   ├── skeleton.py           # 3D skeleton model
│   └── smoothing.py          # Temporal EMA filtering
├── ergonomics/
│   ├── angles.py             # Joint angle computation from 3D bone vectors
│   ├── reba.py               # REBA scoring (5 tables, midpoint ranges)
│   └── risk.py               # Risk level classification + color mapping
├── api/
│   └── server.py             # FastAPI WebSocket server for Web Dashboard
└── feedback/
    └── overlay.py            # Real-time skeleton + score overlay
```

---

## REBA Scoring

The system implements REBA (Rapid Entire Body Assessment) with 5 scoring tables:

| Table | Assessment | Score Range |
|-------|-----------|-------------|
| A | Trunk + Neck + Legs + Load | 1–12 |
| B | Upper Arm + Lower Arm + Wrist + Coupling | 1–12 |
| C | Combined (Table A × Table B) | 1–12 |
| Final | Score C + Activity | 1–15 |

**Risk Levels:** Negligible (1), Low (2–3), Medium (4–7), High (8–10), Very High (11–15)

---

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Planned Phases

- **Phase 1** (✓) — Single camera, single person, REBA scoring, real-time overlay
- **Phase 2** — Multi-person tracking (ByteTrack), multi-camera triangulation, RULA scoring
- **Phase 3** — Web dashboard (FastAPI + Three.js), CSV reporting, alert system
- **Phase 4** — TensorRT optimization, SMPL integration, recommendation engine

---

## License

CC BY-NC (VideoPose3D weights). Project code provided under MIT.
