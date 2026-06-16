from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    source: int | str = 0
    frame_width: int = 640
    frame_height: int = 480

class DetectionConfig(BaseModel):
    model_size: Literal["n", "s", "m", "l", "x"] = "n"
    conf_threshold: float = 0.5
    device: str = "cpu"

class LiftingConfig(BaseModel):
    model_path: str = "models/videopose3d_243.bin"
    temporal_window: int = 243
    device: str = "cpu"

class ErgonomicsConfig(BaseModel):
    reba_threshold: int = Field(default=8, ge=1, le=15)
    smoothing_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    load_weight_kg: float = 0.0

class FeedbackConfig(BaseModel):
    show_fps: bool = True
    show_skeleton: bool = True
    fullscreen: bool = False

class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    stream_fps: int = 30

class AeroPoseConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    lifting: LiftingConfig = LiftingConfig()
    ergonomics: ErgonomicsConfig = ErgonomicsConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    web: WebConfig = WebConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AeroPoseConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
