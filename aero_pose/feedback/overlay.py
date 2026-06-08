import cv2
import numpy as np

from aero_pose.lifting.spatial_vectors import (
    COCO_SKELETON_EDGES,
    H36M_SKELETON_EDGES,
    project_3d_to_viewport,
)
from aero_pose.ergonomics.reba import REBAResult


COCO_REGIONS: dict[str, list[tuple[int, int]]] = {
    "face": [(0, 1), (0, 2), (1, 3), (2, 4)],
    "trunk": [(5, 11), (6, 12), (5, 6)],
    "arms": [(5, 7), (7, 9), (6, 8), (8, 10)],
    "legs": [(11, 13), (13, 15), (12, 14), (14, 16)],
}

H36M_REGIONS: dict[str, list[tuple[int, int]]] = {
    "trunk": [(0, 1), (0, 4), (1, 2), (4, 5), (2, 3), (5, 6)],
    "neck": [(0, 7), (7, 8), (8, 9)],
    "arms": [(8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)],
}

PANEL_ORDER = ["trunk", "neck", "arms", "legs"]


class OverlayRenderer:
    def __init__(self) -> None:
        self._joint_radius = 4
        self._line_thickness = 2
        self._vp_size = 220
        self._auto_rotate_deg: float = 0.0

    def draw(
        self,
        frame: np.ndarray,
        keypoints_2d: np.ndarray | None,
        joints_3d: np.ndarray | None = None,
        reba_result: REBAResult | None = None,
        fps: float | None = None,
        warmup: bool = False,
        warmup_progress: int = 0,
    ) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]

        if warmup:
            self._draw_text(
                display,
                f"Warming up 3D lifter... ({warmup_progress}/243)",
                (w // 2 - 130, 30),
                (255, 255, 255),
                0.7,
            )

        if keypoints_2d is not None and keypoints_2d.shape[0] >= 17:
            kp = keypoints_2d if keypoints_2d.shape[-1] == 2 else keypoints_2d[:, :2]
            self._draw_2d_skeleton(display, kp, reba_result)
        else:
            self._draw_text(
                display,
                "No person detected",
                (w // 2 - 100, h // 2),
                (128, 128, 128),
                0.8,
            )

        if joints_3d is not None:
            self._auto_rotate_deg = (self._auto_rotate_deg + 0.5) % 360
            self._draw_3d_viewport(display, joints_3d, reba_result)

        if reba_result is not None:
            self._draw_reba_badge(display, reba_result)
            self._draw_info_panel(display, reba_result)

        if fps is not None:
            self._draw_text(display, f"FPS: {fps:.1f}", (10, 25),
                            (255, 255, 255), 0.6)

        return display

    def _draw_2d_skeleton(
        self,
        display: np.ndarray,
        kp: np.ndarray,
        reba_result: REBAResult | None,
    ) -> None:
        visible = kp[:, 0] > 0
        colors = reba_result.region_colors if reba_result else {}
        dim = (60, 60, 60)
        dim_body = (80, 80, 80)

        region_color_map = {
            "trunk": colors.get("trunk", dim),
            "neck": colors.get("neck", dim),
            "arms": colors.get("upper_arm", dim),
            "legs": colors.get("legs", dim_body),
            "face": (100, 100, 100),
        }

        for region, edges in COCO_REGIONS.items():
            color = region_color_map.get(region, dim)
            for child, parent in edges:
                if visible[parent] and visible[child]:
                    self._draw_line_transparent(
                        display,
                        (int(kp[parent, 0]), int(kp[parent, 1])),
                        (int(kp[child, 0]), int(kp[child, 1])),
                        color,
                        self._line_thickness,
                    )

        for i in range(len(kp)):
            if visible[i]:
                cx, cy = int(kp[i, 0]), int(kp[i, 1])
                cv2.circle(display, (cx, cy), self._joint_radius,
                           (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(display, (cx, cy), self._joint_radius + 1,
                           (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_line_transparent(
        img: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        overlay = img.copy()
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    def _draw_3d_viewport(
        self,
        display: np.ndarray,
        joints_3d: np.ndarray,
        reba_result: REBAResult | None,
    ) -> None:
        h, w = display.shape[:2]
        vs = self._vp_size
        ox, oy = 10, h - vs - 10

        cv2.rectangle(display, (ox, oy), (ox + vs, oy + vs), (15, 15, 30), -1)
        border_color = reba_result.risk_level.color if reba_result else (80, 80, 80)
        cv2.rectangle(display, (ox, oy), (ox + vs, oy + vs), border_color, 1)

        self._draw_text(display, "3D VIEW", (ox + 5, oy + 15),
                        (180, 180, 180), 0.45)

        az = (-70.0 + self._auto_rotate_deg) % 360
        pts = project_3d_to_viewport(
            joints_3d, vs, elevation_deg=15.0, azimuth_deg=az
        )

        cx, cy = ox + vs // 2, oy + vs // 2
        pts[:, 0] += cx
        pts[:, 1] += cy

        self._draw_floor_grid(display, pts, joints_3d, cx, cy, az)

        colors = reba_result.region_colors if reba_result else {}
        dim = (80, 80, 80)

        for region, edges in H36M_REGIONS.items():
            color = colors.get(region, dim) if region in ("trunk", "neck") else dim
            for child, parent in edges:
                p1 = (int(pts[parent, 0]), int(pts[parent, 1]))
                p2 = (int(pts[child, 0]), int(pts[child, 1]))
                cv2.line(display, p1, p2, color, 2, cv2.LINE_AA)

        for i in range(len(pts)):
            cv2.circle(display, (int(pts[i, 0]), int(pts[i, 1])),
                       3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display, (int(pts[i, 0]), int(pts[i, 1])),
                       4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_floor_grid(
        self,
        display: np.ndarray,
        pts_2d: np.ndarray,
        joints_3d: np.ndarray,
        cx: int,
        cy: int,
        azimuth_deg: float,
    ) -> None:
        vs = self._vp_size
        n = 5
        spacing = 0.2
        coords = np.linspace(-spacing * 2, spacing * 2, n)
        grid_3d = np.zeros((n * n, 3), dtype=np.float32)
        idx = 0
        for x in coords:
            for y in coords:
                grid_3d[idx, 0] = x
                grid_3d[idx, 1] = y
                grid_3d[idx, 2] = joints_3d[0, 2]
                idx += 1

        centroid = joints_3d.mean(axis=0)
        centered = grid_3d - centroid
        a_az = np.radians(azimuth_deg)
        a_el = np.radians(15.0)
        R = np.array([
            [np.cos(a_az), np.sin(a_az) * np.sin(a_el),
             np.sin(a_az) * np.cos(a_el)],
            [0, np.cos(a_el), -np.sin(a_el)],
            [-np.sin(a_az), np.cos(a_az) * np.sin(a_el),
             np.cos(a_az) * np.cos(a_el)],
        ], dtype=np.float32)
        rotated = centered @ R.T
        scale = vs / 3.5
        grid_2d = rotated[:, :2] * scale + np.array([cx, cy])
        grid_pts = grid_2d.reshape(n, n, 2)

        for i in range(n):
            c = (40, 40, 60)
            p1 = tuple(map(int, grid_pts[i, 0]))
            p2 = tuple(map(int, grid_pts[i, -1]))
            cv2.line(display, p1, p2, c, 1, cv2.LINE_AA)
            p1 = tuple(map(int, grid_pts[0, i]))
            p2 = tuple(map(int, grid_pts[-1, i]))
            cv2.line(display, p1, p2, c, 1, cv2.LINE_AA)

    def _draw_info_panel(
        self,
        display: np.ndarray,
        result: REBAResult,
    ) -> None:
        labels = []
        for region in PANEL_ORDER:
            color = result.region_colors.get(region, (100, 100, 100))
            status = result.status_labels.get(region, "?")
            labels.append((region.upper(), status, color))

        px, py = 12, 52
        row_h = 28
        pw = 165
        ph = len(labels) * row_h + 12

        sub = display[py:py + ph, px:px + pw].copy()
        cv2.rectangle(sub, (0, 0), (pw, ph), (0, 0, 0), -1)
        display[py:py + ph, px:px + pw] = cv2.addWeighted(
            display[py:py + ph, px:px + pw], 0.5, sub, 0.5, 0
        )

        for i, (name, status, color) in enumerate(labels):
            y = py + 10 + i * row_h
            cv2.circle(display, (px + 14, y + 4), 6, color, -1, cv2.LINE_AA)
            cv2.circle(display, (px + 14, y + 4), 7, (255, 255, 255), 1, cv2.LINE_AA)
            self._draw_text(display, f"{name:6s} {status}",
                            (px + 28, y + 9), (220, 220, 220), 0.5)

    def _draw_reba_badge(self, display: np.ndarray, result: REBAResult) -> None:
        h, w = display.shape[:2]
        color = result.risk_level.color
        label = result.risk_level.label

        bx = w - 240
        by = 10

        cv2.rectangle(display, (bx, by), (bx + 230, by + 105), (0, 0, 0), -1)
        cv2.rectangle(display, (bx, by), (bx + 230, by + 105), color, 2)

        self._draw_text(display, f"REBA: {result.final_score}",
                        (bx + 10, by + 28), color, 0.85, 2)
        self._draw_text(display, f"Risk: {label}",
                        (bx + 10, by + 50), color, 0.65)
        self._draw_text(display, f"A:{result.score_a} B:{result.score_b} C:{result.score_c}",
                        (bx + 10, by + 72), (200, 200, 200), 0.55)
        self._draw_text(display, f"T:{result.trunk_score} N:{result.neck_score} L:{result.leg_score}",
                        (bx + 10, by + 92), (150, 150, 150), 0.5)

    @staticmethod
    def _draw_text(
        display: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        scale: float,
        thickness: int = 1,
    ) -> None:
        cv2.putText(display, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
