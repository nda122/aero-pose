from dataclasses import dataclass, field

from aero_pose.ergonomics.risk import RiskLevel


@dataclass
class REBAResult:
    score_a: int = 1
    score_b: int = 1
    score_c: int = 1
    final_score: int = 1
    risk_level: RiskLevel = RiskLevel.NEGLIGIBLE
    trunk_score: int = 1
    neck_score: int = 1
    leg_score: int = 1
    upper_arm_score: int = 1
    lower_arm_score: int = 1
    wrist_score: int = 1
    load_score: int = 0
    coupling_score: int = 0
    activity_score: int = 0

    @property
    def region_colors(self) -> dict[str, tuple[int, int, int]]:
        def _score_color(s: int, mid: int = 2) -> tuple[int, int, int]:
            if s <= 1:
                return (0, 255, 0)
            if s <= mid:
                return (0, 255, 255)
            return (0, 0, 255)
        return {
            "trunk": _score_color(self.trunk_score),
            "neck": _score_color(self.neck_score),
            "legs": _score_color(self.leg_score),
            "upper_arm": _score_color(self.upper_arm_score, mid=2),
            "lower_arm": _score_color(self.lower_arm_score),
            "wrist": _score_color(self.wrist_score),
        }

    @property
    def status_labels(self) -> dict[str, str]:
        def _label(s: int, mid: int = 2) -> str:
            if s <= 1:
                return "GOOD"
            if s <= mid:
                return "WARN"
            return "BAD"
        return {
            "trunk": _label(self.trunk_score),
            "neck": _label(self.neck_score),
            "legs": _label(self.leg_score),
            "arms": _label(max(self.upper_arm_score, self.lower_arm_score, self.wrist_score), mid=2),
        }


TRUNK_TABLE = [
    (0, 0, 1),
    (0, 20, 2),
    (20, 60, 3),
    (60, 180, 4),
]

NECK_TABLE = [
    (0, 20, 1),
    (20, 180, 2),
]

def _lookup(angle: float, table: list[tuple[float, float, int]]) -> int:
    for lo, hi, score in table:
        if lo <= angle < hi:
            return score
    return table[-1][2]


class REBAScorer:
    def __init__(self, load_weight_kg: float = 0.0) -> None:
        self._load_weight = load_weight_kg

    def score(self, angles: dict) -> REBAResult:
        trunk = _lookup(angles.get("trunk_flexion", 0), TRUNK_TABLE)
        neck = _lookup(angles.get("neck_flexion", 0), NECK_TABLE)
        legs = self._score_legs(angles)

        score_a = trunk + neck + legs
        load = self._score_load()
        score_a_combined = self._table_a_load(score_a, load)

        ua_l = self._score_upper_arm(angles.get("upper_arm_l", 0))
        ua_r = self._score_upper_arm(angles.get("upper_arm_r", 0))
        upper_arm = max(ua_l, ua_r)

        fa_l = self._score_lower_arm(angles.get("lower_arm_l", 0))
        fa_r = self._score_lower_arm(angles.get("lower_arm_r", 0))
        lower_arm = max(fa_l, fa_r)

        wr_l = self._score_wrist(angles.get("wrist_l", 0))
        wr_r = self._score_wrist(angles.get("wrist_r", 0))
        wrist = max(wr_l, wr_r)

        score_b = upper_arm + lower_arm + wrist
        coupling = 0
        score_b_combined = self._table_b_coupling(score_b, coupling)

        score_c = self._table_c(score_a_combined, score_b_combined)
        activity = 0
        final = score_c + activity

        return REBAResult(
            score_a=score_a_combined,
            score_b=score_b_combined,
            score_c=score_c,
            final_score=final,
            risk_level=RiskLevel.from_reba(final),
            trunk_score=trunk,
            neck_score=neck,
            leg_score=legs,
            upper_arm_score=upper_arm,
            lower_arm_score=lower_arm,
            wrist_score=wrist,
            load_score=load,
            coupling_score=coupling,
            activity_score=activity,
        )

    @staticmethod
    def _score_legs(angles: dict) -> int:
        l_knee = angles.get("knee_l", 0)
        r_knee = angles.get("knee_r", 0)
        max_knee = max(l_knee, r_knee)
        score = 1
        if max_knee > 60:
            score += 2
        elif max_knee > 30:
            score += 1
        return score

    @staticmethod
    def _score_upper_arm(angle: float) -> int:
        if angle < 0 or angle > 180:
            return 1
        if angle > 90:
            return 4
        if angle > 45:
            return 3
        if angle > 20:
            return 2
        return 1

    @staticmethod
    def _score_lower_arm(angle: float) -> int:
        if 60 <= angle <= 100:
            return 1
        return 2

    @staticmethod
    def _score_wrist(angle: float) -> int:
        if angle <= 15:
            return 1
        return 2

    def _score_load(self) -> int:
        w = self._load_weight
        if w < 5:
            return 0
        if w <= 10:
            return 1
        return 2

    @staticmethod
    def _table_a_load(score_a: int, load: int) -> int:
        adjusted = score_a + load
        mapping = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
                   9: 8, 10: 9, 11: 10, 12: 11}
        return mapping.get(adjusted, min(adjusted, 11))

    @staticmethod
    def _table_b_coupling(score_b: int, coupling: int) -> int:
        adjusted = score_b + coupling
        mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                   8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
        return mapping.get(adjusted, min(adjusted, 12))

    @staticmethod
    def _table_c(score_a: int, score_b: int) -> int:
        table_c = [
            [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
            [1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8],
            [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
            [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
            [7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 11],
            [8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11],
            [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        ]
        a_idx = max(0, min(score_a - 1, 11))
        b_idx = max(0, min(score_b - 1, 11))
        return table_c[a_idx][b_idx]
