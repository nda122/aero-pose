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
        self._vp_size = 180
        self.azimuth: float = 35.0
        self.elevation: float = 15.0
        self._panel_w = 300 # Increased panel width for better spacing

    def draw(
        self,
        frame: np.ndarray,
        keypoints_2d: np.ndarray | None,
        joints_3d: np.ndarray | None = None,
        reba_result: REBAResult | None = None,
        fps: float | None = None,
        warmup: bool = False,
        warmup_progress: int = 0,
        azimuth: float | None = None,
        elevation: float | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        # Create new canvas: Camera on the left + Panel on the right
        display = np.zeros((h, w + self._panel_w, 3), dtype=np.uint8)
        
        cam_view = frame.copy()

        if warmup:
            self._draw_text(
                cam_view,
                f"Warming up 3D lifter... ({warmup_progress}/243)",
                (w // 2 - 140, 40),
                (255, 255, 255),
                0.5,
                1
            )

        if keypoints_2d is not None and keypoints_2d.shape[0] >= 17:
            kp = keypoints_2d if keypoints_2d.shape[-1] == 2 else keypoints_2d[:, :2]
            self._draw_2d_skeleton(cam_view, kp, reba_result)
        else:
            self._draw_text(
                cam_view,
                "No person detected",
                (w // 2 - 100, h // 2),
                (128, 128, 128),
                0.8,
            )

        display[:, :w] = cam_view # Place camera view on the left

        if joints_3d is not None:
            if azimuth is not None:
                self.azimuth = azimuth
            if elevation is not None:
                self.elevation = elevation
                
            self._draw_3d_viewport(display, joints_3d, reba_result)

        if reba_result is not None:
            self._draw_reba_badge(display, reba_result)

        if fps is not None:
            # Draw FPS box at the bottom-left of the camera view
            cv2.rectangle(display, (10, h - 35), (120, h - 10), (20, 20, 20), -1) # Dark background
            cv2.rectangle(display, (10, h - 35), (120, h - 10), (76, 175, 80), 1) # Green border
            self._draw_text(display, f"FPS: {int(fps)}", (20, h - 17), # Text position
                            (76, 175, 80), 0.45, 1)

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
        h, full_w = display.shape[:2]
        w = full_w - self._panel_w
        vs = self._vp_size

        ox = w + (self._panel_w - vs) // 2
        oy = 265

        cv2.rectangle(display, (ox, oy), (ox + vs, oy + vs), (10, 10, 20), -1)
        color = reba_result.risk_level.color if reba_result else (100, 100, 100)
        cv2.rectangle(display, (ox, oy), (ox + vs, oy + vs), color, 1, cv2.LINE_AA)

        self._draw_text(display, "3D SPATIAL RECONSTRUCTION", (ox, oy - 10),
                        (180, 180, 180), 0.35)

        pts = project_3d_to_viewport(
            joints_3d, vs, elevation_deg=self.elevation, azimuth_deg=self.azimuth
        )

        cx, cy = ox + vs // 2, oy + vs // 2
        pts[:, 0] += cx
        pts[:, 1] += cy

        self._draw_floor_grid(display, pts, joints_3d, cx, cy, self.azimuth, self.elevation)
        
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
        elevation_deg: float,
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
        a_el = np.radians(elevation_deg)
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
        h, full_w = display.shape[:2]
        w = full_w - self._panel_w
        color = result.risk_level.color
        label = result.risk_level.label
        
        # Position at the top of the right panel
        bx = w + 20 # X-offset from camera view width
        by = 20 # Y-offset from top

        self._draw_text(display, "REAL-TIME ERGONOMIC MONITOR", (bx, by), (150, 150, 150), 0.35)
        self._draw_text(display, "AERO-POSE", (bx, by + 20), (255, 255, 255), 0.5, 2) # Main title

        # REBA Score (Main Badge)
        score_y = by + 75 # Y position for the score
        self._draw_text(display, f"{result.final_score}", (bx + 40, score_y), color, 2.0, 3) # Score text
        
        # Risk Label
        label_y = score_y + 30 # Y position for the risk label
        cv2.rectangle(display, (bx, label_y), (bx + self._panel_w - 40, label_y + 20), color, -1) # Background rectangle
        self._draw_text(display, label, (bx + 12, label_y + 15), (255, 255, 255), 0.38, 2) # Risk label text

        # Detailed scores (Breakdown)
        dy = label_y + 50 # Starting Y position for breakdown details
        self._draw_text(display, "SYSTEM BREAKDOWN", (bx, dy - 10), (130, 130, 130), 0.35) # Breakdown title
        
        # Group A
        self._draw_text(display, f"TRUNK:{result.trunk_score} NECK:{result.neck_score} LEG:{result.leg_score}", 
                        (bx, dy + 12), (200, 200, 200), 0.38) # Group A scores
        # Group B
        self._draw_text(display, f"U-ARM:{result.upper_arm_score} L-ARM:{result.lower_arm_score} WRIST:{result.wrist_score}", 
                        (bx, dy + 28), (200, 200, 200), 0.38) # Group B scores
        # Tables
        self._draw_text(display, f"TABLES A:{result.score_a} B:{result.score_b} C:{result.score_c}", 
                        (bx, dy + 48), (76, 175, 80), 0.38) # Table scores

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
