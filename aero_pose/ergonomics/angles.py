import numpy as np

from aero_pose.lifting.spatial_vectors import bone_vector, angle_between


def acute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    a = angle_between(v1, v2)
    return min(a, 180.0 - a)


def compute_joint_angles(joints: np.ndarray) -> dict:
    up = np.array([0, 0, 1], dtype=np.float32)
    fwd = np.array([0, 1, 0], dtype=np.float32)

    neck = joints[8]
    head = joints[9]
    hip = joints[0]
    l_shoulder = joints[11]
    r_shoulder = joints[14]
    l_elbow = joints[12]
    r_elbow = joints[15]
    l_wrist = joints[13]
    r_wrist = joints[16]
    l_hip = joints[4]
    r_hip = joints[1]
    l_knee = joints[5]
    r_knee = joints[2]
    l_ankle = joints[6]
    r_ankle = joints[3]

    trunk_vec = neck - hip
    trunk_angle = angle_between(trunk_vec, up) if np.linalg.norm(trunk_vec) > 1e-8 else 0.0

    neck_vec = head - neck
    neck_angle = acute_angle(neck_vec, trunk_vec) if (
        np.linalg.norm(neck_vec) > 1e-8 and np.linalg.norm(trunk_vec) > 1e-8
    ) else 0.0

    l_upper_arm = l_elbow - l_shoulder
    l_ua_angle = acute_angle(l_upper_arm, trunk_vec) if (
        np.linalg.norm(l_upper_arm) > 1e-8 and np.linalg.norm(trunk_vec) > 1e-8
    ) else 0.0

    r_upper_arm = r_elbow - r_shoulder
    r_ua_angle = acute_angle(r_upper_arm, trunk_vec) if (
        np.linalg.norm(r_upper_arm) > 1e-8 and np.linalg.norm(trunk_vec) > 1e-8
    ) else 0.0

    l_forearm = l_wrist - l_elbow
    l_fa_angle = angle_between(l_forearm, l_upper_arm) if (
        np.linalg.norm(l_forearm) > 1e-8 and np.linalg.norm(l_upper_arm) > 1e-8
    ) else 0.0

    r_forearm = r_wrist - r_elbow
    r_fa_angle = angle_between(r_forearm, r_upper_arm) if (
        np.linalg.norm(r_forearm) > 1e-8 and np.linalg.norm(r_upper_arm) > 1e-8
    ) else 0.0

    l_wrist_vec = l_wrist - (l_elbow + l_wrist) / 2
    r_wrist_vec = r_wrist - (r_elbow + r_wrist) / 2
    l_wrist_angle = angle_between(l_wrist_vec, l_forearm) if (
        np.linalg.norm(l_wrist_vec) > 1e-8 and np.linalg.norm(l_forearm) > 1e-8
    ) else 0.0
    r_wrist_angle = angle_between(r_wrist_vec, r_forearm) if (
        np.linalg.norm(r_wrist_vec) > 1e-8 and np.linalg.norm(r_forearm) > 1e-8
    ) else 0.0

    l_knee_vec = l_ankle - l_knee
    l_knee_angle = angle_between(l_knee_vec, l_knee - l_hip) if (
        np.linalg.norm(l_knee_vec) > 1e-8 and np.linalg.norm(l_knee - l_hip) > 1e-8
    ) else 0.0

    r_knee_vec = r_ankle - r_knee
    r_knee_angle = angle_between(r_knee_vec, r_knee - r_hip) if (
        np.linalg.norm(r_knee_vec) > 1e-8 and np.linalg.norm(r_knee - r_hip) > 1e-8
    ) else 0.0

    return {
        "trunk_flexion": trunk_angle,
        "neck_flexion": neck_angle,
        "upper_arm_l": l_ua_angle,
        "upper_arm_r": r_ua_angle,
        "lower_arm_l": l_fa_angle,
        "lower_arm_r": r_fa_angle,
        "wrist_l": l_wrist_angle,
        "wrist_r": r_wrist_angle,
        "knee_l": l_knee_angle,
        "knee_r": r_knee_angle,
    }
