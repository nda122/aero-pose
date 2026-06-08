import numpy as np

from aero_pose.ergonomics.angles import compute_joint_angles
from aero_pose.ergonomics.reba import REBAScorer
from aero_pose.ergonomics.risk import RiskLevel


def _standing_pose() -> np.ndarray:
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[0] = [0, 0, 0]
    joints[1] = [-0.1, 0, 0]
    joints[4] = [0.1, 0, 0]
    joints[2] = [-0.1, 0, -0.5]
    joints[5] = [0.1, 0, -0.5]
    joints[3] = [-0.1, 0, -0.9]
    joints[6] = [0.1, 0, -0.9]
    joints[7] = [0, 0, 0.3]
    joints[8] = [0, 0, 0.6]
    joints[9] = [0, 0, 0.75]
    joints[10] = [0, 0, 0.8]
    joints[11] = [-0.2, 0.1, 0.55]
    joints[12] = [-0.35, 0.15, 0.45]
    joints[13] = [-0.45, 0.2, 0.35]
    joints[14] = [0.2, 0.1, 0.55]
    joints[15] = [0.35, 0.15, 0.45]
    joints[16] = [0.45, 0.2, 0.35]
    return joints


def _bent_pose() -> np.ndarray:
    joints = np.zeros((17, 3), dtype=np.float32)
    joints[0] = [0, 0.3, 0]
    joints[1] = [-0.1, 0.25, 0]
    joints[4] = [0.1, 0.25, 0]
    joints[2] = [-0.1, -0.2, 0]
    joints[5] = [0.1, -0.2, 0]
    joints[3] = [-0.1, -0.6, 0]
    joints[6] = [0.1, -0.6, 0]
    joints[7] = [0, 0.2, 0.3]
    joints[8] = [0, 0.1, 0.5]
    joints[9] = [0, 0.05, 0.6]
    joints[10] = [0, 0, 0.65]
    joints[11] = [-0.3, 0.05, 0.45]
    joints[12] = [-0.5, -0.05, 0.3]
    joints[13] = [-0.6, -0.1, 0.2]
    joints[14] = [0.3, 0.05, 0.45]
    joints[15] = [0.5, -0.05, 0.3]
    joints[16] = [0.6, -0.1, 0.2]
    return joints


def test_standing_trunk_angle() -> None:
    joints = _standing_pose()
    angles = compute_joint_angles(joints)
    assert angles["trunk_flexion"] < 15, (
        f"Standing trunk should be near 0, got {angles['trunk_flexion']}"
    )


def test_bent_trunk_angle() -> None:
    joints = _bent_pose()
    angles = compute_joint_angles(joints)
    assert angles["trunk_flexion"] > 15, (
        f"Bent trunk should be > 15, got {angles['trunk_flexion']}"
    )


def test_standing_reba_score() -> None:
    joints = _standing_pose()
    angles = compute_joint_angles(joints)
    scorer = REBAScorer()
    result = scorer.score(angles)
    assert result.final_score <= 6, (
        f"Standing REBA should be low (<=6), got {result.final_score}"
    )
    assert result.trunk_score <= 2
    assert result.leg_score == 1


def test_bent_reba_score() -> None:
    standing = _standing_pose()
    bent = _bent_pose()
    scorer = REBAScorer()
    standing_result = scorer.score(compute_joint_angles(standing))
    bent_result = scorer.score(compute_joint_angles(bent))
    assert bent_result.final_score > standing_result.final_score, (
        f"Bent REBA ({bent_result.final_score}) should be > Standing ({standing_result.final_score})"
    )
    assert bent_result.trunk_score > standing_result.trunk_score, (
        f"Bent trunk ({bent_result.trunk_score}) should be > Standing trunk ({standing_result.trunk_score})"
    )


def test_risk_level_mapping() -> None:
    assert RiskLevel.from_reba(1) == RiskLevel.NEGLIGIBLE
    assert RiskLevel.from_reba(2) == RiskLevel.LOW
    assert RiskLevel.from_reba(3) == RiskLevel.LOW
    assert RiskLevel.from_reba(4) == RiskLevel.MEDIUM
    assert RiskLevel.from_reba(7) == RiskLevel.MEDIUM
    assert RiskLevel.from_reba(8) == RiskLevel.HIGH
    assert RiskLevel.from_reba(10) == RiskLevel.HIGH
    assert RiskLevel.from_reba(11) == RiskLevel.VERY_HIGH
    assert RiskLevel.from_reba(15) == RiskLevel.VERY_HIGH


def test_reba_table_c_values() -> None:
    scorer = REBAScorer()
    angles = _standing_pose()
    result = scorer.score(compute_joint_angles(angles))
    assert 1 <= result.score_a <= 12
    assert 1 <= result.score_b <= 12
    assert 1 <= result.score_c <= 12
    assert 1 <= result.final_score <= 15
