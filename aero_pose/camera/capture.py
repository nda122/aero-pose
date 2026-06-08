from pathlib import Path
import cv2
import numpy as np


class CameraCapture:
    def __init__(self, source: int | str | Path) -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {source}")
        self._source = source

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read(self) -> tuple[bool, np.ndarray]:
        return self._cap.read()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._cap.release()

    def set_frame_size(self, width: int, height: int) -> None:
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, *args) -> None:
        self.release()
