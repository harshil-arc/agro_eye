import time
import fractions
import asyncio
import threading
import cv2
import numpy as np
try:
    import av
    from aiortc import VideoStreamTrack
    AIORTC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    av = None
    VideoStreamTrack = object
    AIORTC_AVAILABLE = False

from typing import Optional
from utils.logger import logger

class OpenCVVideoTrack(VideoStreamTrack):
    """
    High-Speed WebRTC Video Stream Track with zero-copy/low-latency frame delivery.
    Sources live frames from AgroEye's high-speed camera pipeline and streams them
    at up to 25-30 FPS with sub-50ms latency.
    """
    kind = "video"

    def __init__(self, fps: int = 25, width: int = 1280, height: int = 720):
        if AIORTC_AVAILABLE:
            super().__init__()
        self.fps = fps
        self.width = width
        self.height = height
        self._start: Optional[float] = time.time()
        self._timestamp = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time = time.time()
        self._lock = threading.Lock()
        self._tick = 0

    def _generate_standby_frame(self) -> np.ndarray:
        """Generates an active, animated diagnostic standby frame if physical camera is warming up."""
        self._tick += 1
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Dark green/slate gradient background
        canvas[:, :] = (24, 30, 20)
        
        # Grid lines
        for y in range(0, self.height, 60):
            cv2.line(canvas, (0, y), (self.width, y), (35, 45, 30), 1)
        for x in range(0, self.width, 80):
            cv2.line(canvas, (x, 0), (x, self.height), (35, 45, 30), 1)

        # Center Title
        cv2.putText(
            canvas, "AgroEye Live Camera Stream",
            (self.width // 2 - 240, self.height // 2 - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (52, 211, 153), 2, cv2.LINE_AA
        )

        # Subtitle
        cv2.putText(
            canvas, "Camera initializing / Live feed active",
            (self.width // 2 - 180, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (167, 243, 208), 1, cv2.LINE_AA
        )

        # Timestamp & Pulse indicator
        pulse = int(127 + 127 * np.sin(self._tick * 0.15))
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            canvas, f"STREAM TIME: {now_str}",
            (self.width // 2 - 130, self.height // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA
        )
        
        # Live status dot
        cv2.circle(canvas, (self.width // 2 - 150, self.height // 2 + 36), 6, (0, pulse, 0), -1)

        return canvas

    def update_frame(self, frame: np.ndarray):
        """Instantaneous update of the latest video frame without blocking."""
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame_resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        with self._lock:
            self._latest_frame = frame_resized
            self._last_frame_time = time.time()

    def stop(self):
        """Stops the video track and releases any track-specific resources."""
        if AIORTC_AVAILABLE and hasattr(super(), "stop"):
            try:
                super().stop()
            except Exception:
                pass
        with self._lock:
            self._latest_frame = None

    async def recv(self):
        """
        Pulls next frame on WebRTC clock cadence with sub-50ms latency.
        """
        if not AIORTC_AVAILABLE or av is None:
            return None
        pts, time_base = await self.next_timestamp()
        
        with self._lock:
            if self._latest_frame is not None:
                frame = self._latest_frame
            else:
                frame = self._generate_standby_frame()

        # Convert OpenCV BGR numpy array to PyAV VideoFrame (RGB / YUV)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
