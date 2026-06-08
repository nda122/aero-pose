import numpy as np


COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

H36M_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


H36M_JOINT_NAMES = [
    "hip", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "spine", "neck", "head", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]

H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5,
                0, 7, 8, 9,
                8, 11, 12, 8, 14, 15]

COCO_INDEX = {name: i for i, name in enumerate([
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
])}


def coco_to_h36m(kp_coco: np.ndarray) -> np.ndarray:
    if kp_coco.shape[-1] == 3:
        kp_coco = kp_coco[..., :2]

    h36m = np.zeros((17, 2), dtype=np.float32)

    ci = COCO_INDEX

    l_hip = kp_coco[ci["l_hip"]]
    r_hip = kp_coco[ci["r_hip"]]
    l_shoulder = kp_coco[ci["l_shoulder"]]
    r_shoulder = kp_coco[ci["r_shoulder"]]

    hip_mid = (l_hip + r_hip) / 2.0
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    spine_mid = (hip_mid + shoulder_mid) / 2.0

    h36m[0] = hip_mid
    h36m[1] = r_hip
    h36m[2] = kp_coco[ci["r_knee"]]
    h36m[3] = kp_coco[ci["r_ankle"]]
    h36m[4] = l_hip
    h36m[5] = kp_coco[ci["l_knee"]]
    h36m[6] = kp_coco[ci["l_ankle"]]
    h36m[7] = spine_mid
    h36m[8] = shoulder_mid
    h36m[9] = kp_coco[ci["nose"]]
    h36m[10] = kp_coco[ci["nose"]] + np.array([0, -10], dtype=np.float32)
    h36m[11] = l_shoulder
    h36m[12] = kp_coco[ci["l_elbow"]]
    h36m[13] = kp_coco[ci["l_wrist"]]
    h36m[14] = r_shoulder
    h36m[15] = kp_coco[ci["r_elbow"]]
    h36m[16] = kp_coco[ci["r_wrist"]]

    return h36m


def normalize_screen_coordinates(X: np.ndarray, w: int, h: int) -> np.ndarray:
    assert X.shape[-1] == 2
    return X / w * 2 - np.array([1, h / w], dtype=np.float32)


def image_coordinates(X: np.ndarray, w: int, h: int) -> np.ndarray:
    assert X.shape[-1] == 2
    return (X + np.array([1, h / w], dtype=np.float32)) * w / 2


def bone_vector(joints: np.ndarray, i: int, j: int) -> np.ndarray:
    return joints[j] - joints[i]


def bone_length(joints: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(bone_vector(joints, i, j)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-8:
        return 0.0
    cos_ang = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def rotation_x(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)


def rotation_y(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def project_3d_to_viewport(
    joints_3d: np.ndarray,
    viewport_size: int,
    elevation_deg: float = 25.0,
    azimuth_deg: float = 35.0,
    scale_factor: float = 3.5,
) -> np.ndarray:
    centroid = joints_3d.mean(axis=0)
    centered = joints_3d - centroid
    R = rotation_y(azimuth_deg) @ rotation_x(elevation_deg)
    rotated = centered @ R.T
    scale = viewport_size / scale_factor
    return rotated[:, :2] * scale
