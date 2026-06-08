import numpy as np
from ultralytics import YOLO

COCO_KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

class YOLOPoseDetector:
    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        model_name = f"yolov8{model_size}-pose.pt"
        self._model = YOLO(model_name)
        self._conf = conf_threshold
        self._device = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self._model(
            frame,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        detections = []
        if not results or results[0].keypoints is None:
            return detections

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        if boxes is None or keypoints is None:
            return detections

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            kp = keypoints.xy[i].cpu().numpy()
            kp_conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else np.ones(17)
            kp_combined = np.column_stack([kp, kp_conf])
            detections.append({
                "bbox": xyxy,
                "keypoints": kp_combined,
                "confidence": conf,
            })
        return detections
