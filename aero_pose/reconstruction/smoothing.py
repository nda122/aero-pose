import numpy as np


class TemporalSmoother:
    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._state: np.ndarray | None = None

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    def reset(self) -> None:
        self._state = None

    def update(self, joints: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = joints.copy()
        else:
            self._state = self._alpha * joints + (1 - self._alpha) * self._state
        return self._state.copy()
