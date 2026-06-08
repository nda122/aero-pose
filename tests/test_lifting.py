import numpy as np

from aero_pose.lifting.spatial_vectors import (
    coco_to_h36m,
    normalize_screen_coordinates,
    image_coordinates,
    H36M_JOINT_NAMES,
)


def test_coco_to_h36m_shape() -> None:
    kp = np.random.randn(17, 2).astype(np.float32)
    h36m = coco_to_h36m(kp)
    assert h36m.shape == (17, 2), f"Expected (17,2), got {h36m.shape}"


def test_coco_to_h36m_hip_midpoint() -> None:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[11] = [-0.2, 0, 1]  # l_hip
    kp[12] = [0.2, 0, 1]   # r_hip
    kp[5] = [0, 0, 0]      # l_shoulder
    kp[6] = [0, 0, 0]      # r_shoulder
    kp[0] = [0, 0, 0]      # nose

    h36m = coco_to_h36m(kp)
    assert abs(h36m[0, 0]) < 0.01, f"Hip midpoint x should be ~0, got {h36m[0, 0]}"


def test_normalize_screen_coordinates() -> None:
    kp = np.array([[320, 240]], dtype=np.float32)
    normalized = normalize_screen_coordinates(kp, w=640, h=480)
    assert abs(normalized[0, 0]) < 0.01, f"Center x should be ~0, got {normalized[0, 0]}"


def test_normalize_roundtrip() -> None:
    kp = np.array([[100, 200]], dtype=np.float32)
    normalized = normalize_screen_coordinates(kp, w=640, h=480)
    restored = image_coordinates(normalized, w=640, h=480)
    assert np.allclose(kp, restored, atol=1e-4), (
        f"Roundtrip failed: {kp} -> {normalized} -> {restored}"
    )


def test_h36m_joint_names_count() -> None:
    assert len(H36M_JOINT_NAMES) == 17, (
        f"Expected 17 H36M joints, got {len(H36M_JOINT_NAMES)}"
    )


def test_coco_to_h36m_with_confidence() -> None:
    kp = np.random.randn(17, 3).astype(np.float32)
    h36m = coco_to_h36m(kp)
    assert h36m.shape == (17, 2), (
        f"Should handle 3-channel (x,y,conf) input, got {h36m.shape}"
    )
