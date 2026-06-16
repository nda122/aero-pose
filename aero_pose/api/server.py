import asyncio
import json
import cv2
import time 
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
from aero_pose.config.settings import AeroPoseConfig
from aero_pose.camera.capture import CameraCapture
from aero_pose.detection.yolo_pose import YOLOPoseDetector
from aero_pose.lifting.single_view import SingleViewLifter
from aero_pose.lifting.spatial_vectors import coco_to_h36m
from aero_pose.reconstruction.smoothing import TemporalSmoother
from aero_pose.ergonomics.angles import compute_joint_angles
from aero_pose.ergonomics.reba import REBAScorer
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background processing loop
    task = asyncio.create_task(processing_loop())
    yield
    # Cancel task when close window
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

FRONTEND_PATH = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"

# Initialize engines
config = AeroPoseConfig()
camera = CameraCapture(config.camera.source)
camera.set_frame_size(1280, 720)

print("Camera Real Size:", 
      camera._cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
      camera._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = YOLOPoseDetector()
lifter = SingleViewLifter(model_path="models/videopose3d_243.bin")
reba_scorer = REBAScorer(load_weight_kg=config.ergonomics.load_weight_kg)
smoother = TemporalSmoother(alpha=config.ergonomics.smoothing_alpha)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    def rgb_to_hex(self, rgb_tuple):
        """Chuyển đổi (B, G, R) sang Hex"""
        return '#%02x%02x%02x' % (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.active_connections)}")
        
    async def broadcast(self, message: str):
        if not self.active_connections:
            return
            
        # Send simultaneously to clients
        tasks = [connection.send_text(message) for connection in self.active_connections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Remove error connections
        for connection, result in zip(self.active_connections, results):
            if isinstance(result, Exception):
                self.disconnect(connection)

manager = ConnectionManager()

def process_heavy_ai_stuff(frame):
    # Encode JPEG
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode("utf-8")
    
    data_payload = None
    detections = detector.detect(frame)
    
    if detections:
        kp_2d = detections[0]['keypoints']
        
        # Initialize data
        data_payload = {
            "kp_2d": kp_2d[:, :2].tolist(),
            "kp_3d": None,
            "reba": None
        }

        kp_h36m = coco_to_h36m(kp_2d[:, :2])
        kp_3d_raw = lifter.lift(kp_h36m)
        
        if kp_3d_raw is not None:
            kp_3d = smoother.update(kp_3d_raw)
            angles = compute_joint_angles(kp_3d)
            reba_res = reba_scorer.score(angles)

            data_payload["kp_3d"] = kp_3d.tolist()
            data_payload["reba"] = {
                "score": reba_res.final_score,
                "risk": reba_res.risk_level.label,
                "color": manager.rgb_to_hex(reba_res.risk_level.color),
                "breakdown": {
                    "trunk": reba_res.trunk_score,
                    "neck": reba_res.neck_score,
                    "legs": reba_res.leg_score,
                    "upper_arm": reba_res.upper_arm_score,
                    "lower_arm": reba_res.lower_arm_score,
                    "wrist": reba_res.wrist_score,
                    "score_a": reba_res.score_a,
                    "score_b": reba_res.score_b,
                    "score_c": reba_res.score_c
                }
            }
            
    return frame_base64, data_payload, frame.shape[1], frame.shape[0]

async def processing_loop():
    # Main loop
    prev_time = time.perf_counter()
    target_fps = getattr(config.web, 'stream_fps', 30) # Default to 30
    frame_delay = 1.0 / target_fps

    while True:
        start_time = time.perf_counter()
        try:
            # Disconnect AI process if no one connected
            if not manager.active_connections:
                await asyncio.sleep(0.5)
                continue

            ret, frame = await asyncio.to_thread(camera.read)
            if not ret:
                await asyncio.sleep(0.03)
                continue

            # Push heavy tasks ThreadPool
            frame_base64, data_payload, width, height = await asyncio.to_thread(process_heavy_ai_stuff, frame)

            # Calculate FPS
            curr_time = time.perf_counter()
            actual_fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            payload = {
                "frame": frame_base64,
                "data": data_payload,
                "fps": round(actual_fps, 1),
                "frame_w": width,
                "frame_h": height,
                "frame_time": time.time()
            }

            asyncio.create_task(manager.broadcast(json.dumps(payload)))

            elapsed = time.perf_counter() - start_time
            wait_time = max(0.001, frame_delay - elapsed)
            await asyncio.sleep(wait_time)

        except Exception as e:
            print("Processing Loop Error:", e)
            await asyncio.sleep(1)

@app.get("/")
async def get():
    return FileResponse(FRONTEND_PATH)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Stay connected for message from clients
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "aero_pose.api.server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )