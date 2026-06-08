from enum import IntEnum


class RiskLevel(IntEnum):
    NEGLIGIBLE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

    @classmethod
    def from_reba(cls, score: int) -> "RiskLevel":
        if score <= 1:
            return cls.NEGLIGIBLE
        if score <= 3:
            return cls.LOW
        if score <= 7:
            return cls.MEDIUM
        if score <= 10:
            return cls.HIGH
        return cls.VERY_HIGH

    @property
    def color(self) -> tuple[int, int, int]:
        return {
            RiskLevel.NEGLIGIBLE: (0, 255, 0),
            RiskLevel.LOW: (0, 255, 255),
            RiskLevel.MEDIUM: (0, 165, 255),
            RiskLevel.HIGH: (0, 0, 255),
            RiskLevel.VERY_HIGH: (0, 0, 128),
        }[self]

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()
