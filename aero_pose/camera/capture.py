from pathlib import Path
import cv2
import numpy as np
import sys


class CameraCapture:
    def __init__(self, source: int | str | Path) -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        # On Windows, the default MSMF backend often encounters 'can't grab frame' errors.
        # Forcing DirectShow (CAP_DSHOW) usually resolves these hardware synchronization issues.
        if isinstance(source, int) and sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            # Fallback to default backend if DSHOW fails to initialize
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(source)
        else:
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
