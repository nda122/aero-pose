class PersonTracker:
    def __init__(self, max_gap_frames: int = 5) -> None:
        self._current: dict | None = None
        self._gap: int = 0
        self._max_gap = max_gap_frames
        self._next_id: int = 0

    def update(self, detections: list[dict]) -> dict | None:
        if not detections:
            self._gap += 1
            if self._gap > self._max_gap:
                self._current = None
            return None

        best = max(detections, key=lambda d: d["confidence"])
        self._gap = 0
        if self._current is None:
            self._current = best.copy()
            self._current["track_id"] = self._next_id
            self._next_id += 1
        else:
            best["track_id"] = self._current["track_id"]
            self._current = best
        return self._current

    def reset(self) -> None:
        self._current = None
        self._gap = 0
