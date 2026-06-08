import sys
import numpy as np
import torch

from aero_pose.lifting.model import TemporalModelOptimized1f
from aero_pose.lifting.spatial_vectors import normalize_screen_coordinates


class SingleViewLifter:
    def __init__(
        self,
        model_path: str,
        window_size: int = 243,
        device: str = "cpu",
        inference_stride: int = 5,
    ) -> None:
        self._window_size = window_size
        self._stride = inference_stride
        self._device = torch.device(device)
        self._buffer: list[np.ndarray] = []
        self._last_output: np.ndarray | None = None
        self._frame_counter: int = 0

        self._model = TemporalModelOptimized1f(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=(3, 3, 3, 3, 3),
            causal=False,
            dropout=0.25,
            channels=1024,
        )
        state = torch.load(model_path, map_location=self._device, weights_only=True)
        if "model_pos" in state:
            state = state["model_pos"]
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    @property
    def is_warmed_up(self) -> bool:
        return len(self._buffer) >= self._window_size

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._last_output = None
        self._frame_counter = 0

    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray | None:
        self._buffer.append(keypoints_2d.copy())
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_size:
            return None

        self._frame_counter += 1
        if self._frame_counter % self._stride != 0:
            return self._last_output

        clip = np.stack(self._buffer[-self._window_size:], axis=0)
        clip_norm = normalize_screen_coordinates(clip, w=640, h=480)

        inp = torch.from_numpy(clip_norm).float().unsqueeze(0)
        inp = inp.to(self._device)

        try:
            with torch.no_grad():
                out = self._model(inp)
            self._last_output = out[0, 0].cpu().numpy()
        except Exception as e:
            print(f"[AERO-POSE] 3D lifting error: {e}", file=sys.stderr)

        return self._last_output
