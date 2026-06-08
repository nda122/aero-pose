import time
from pathlib import Path

import cv2
import typer

from aero_pose.config.settings import AeroPoseConfig
from aero_pose.camera.capture import CameraCapture
from aero_pose.detection.yolo_pose import YOLOPoseDetector
from aero_pose.tracking.tracker import PersonTracker
from aero_pose.lifting.single_view import SingleViewLifter
from aero_pose.lifting.spatial_vectors import coco_to_h36m
from aero_pose.reconstruction.smoothing import TemporalSmoother
from aero_pose.ergonomics.angles import compute_joint_angles
from aero_pose.ergonomics.reba import REBAScorer
from aero_pose.feedback.overlay import OverlayRenderer

app = typer.Typer()


def _resolve_model_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    alt = Path(__file__).parent.parent / path
    if alt.exists():
        return alt
    alt2 = Path(__file__).parent.parent / "models" / "videopose3d_243.bin"
    if alt2.exists():
        return alt2
    return p


@app.command()
def run(
    config: str = typer.Option("config.yaml", "--config", "-c",
                               help="Path to YAML config file"),
    source: str = typer.Option(None, "--source", "-s",
                               help="Camera index or video file path"),
    model_size: str = typer.Option(None, "--model", "-m",
                                   help="YOLOv8 pose model size (n/s/m/l/x)"),
) -> None:
    cfg = AeroPoseConfig.from_yaml(config)

    if source is not None:
        cfg.camera.source = int(source) if source.isdigit() else source
    if model_size is not None:
        cfg.detection.model_size = model_size

    model_path = _resolve_model_path(cfg.lifting.model_path)
    if not model_path.exists():
        typer.echo(
            f"Error: VideoPose3D model not found at {model_path}\n"
            "Download it first:\n"
            "  wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin "
            "-O models/videopose3d_243.bin",
            err=True,
        )
        raise typer.Exit(1)

    try:
        camera = CameraCapture(cfg.camera.source)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if cfg.camera.frame_width and cfg.camera.frame_height:
        camera.set_frame_size(cfg.camera.frame_width, cfg.camera.frame_height)

    detector = YOLOPoseDetector(
        model_size=cfg.detection.model_size,
        conf_threshold=cfg.detection.conf_threshold,
        device=cfg.detection.device,
    )
    tracker = PersonTracker()
    lifter = SingleViewLifter(
        model_path=str(model_path),
        window_size=cfg.lifting.temporal_window,
        device=cfg.lifting.device,
        inference_stride=5,
    )
    smoother = TemporalSmoother(alpha=cfg.ergonomics.smoothing_alpha)
    reba_scorer = REBAScorer(load_weight_kg=cfg.ergonomics.load_weight_kg)
    overlay = OverlayRenderer()

    typer.echo(f"AERO-POSE started. Camera: {cfg.camera.source}")
    typer.echo(f"Warming up 3D lifter ({cfg.lifting.temporal_window} frames)...")

    frame_count = 0
    warmup = True
    last_warmup_frame = 0

    with camera:
        while camera.is_opened():
            t_start = time.perf_counter()

            ret, frame = camera.read()
            if not ret:
                typer.echo("End of video stream.", err=True)
                break

            detections = detector.detect(frame)
            person = tracker.update(detections)

            kp_2d = None
            kp_3d = None
            reba_result = None

            if person is not None:
                kp_2d = person["keypoints"]
                kp_coco = kp_2d[:, :2]
                kp_h36m = coco_to_h36m(kp_coco)

                kp_3d_raw = lifter.lift(kp_h36m)
                if kp_3d_raw is not None:
                    kp_3d = smoother.update(kp_3d_raw)
                    angles = compute_joint_angles(kp_3d)
                    reba_result = reba_scorer.score(angles)
                    warmup = False
                    last_warmup_frame = frame_count

            t_elapsed = time.perf_counter() - t_start
            fps = 1.0 / t_elapsed if t_elapsed > 0 else 0

            display = overlay.draw(
                frame,
                kp_2d,
                joints_3d=kp_3d,
                reba_result=reba_result,
                fps=fps,
                warmup=warmup,
                warmup_progress=lifter.buffer_size,
            )

            cv2.imshow("AERO-POSE", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_count += 1

    cv2.destroyAllWindows()
    typer.echo(f"Session ended. Processed {frame_count} frames.")


@app.command()
def calibrate(
    source: int = typer.Argument(0, help="Camera index"),
    pattern_size: str = typer.Option("9x6", "--pattern", "-p",
                                     help="Checkerboard pattern (cols x rows)"),
) -> None:
    typer.echo("Camera calibration is planned for Phase 2.")
    typer.echo(f"Would calibrate camera {source} with {pattern_size} checkerboard.")


if __name__ == "__main__":
    app()
