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

    def __init__(self, fps: int = 25, width: int = 640, height: int = 480):
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
        
        # Standby placeholder
        self._placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            self._placeholder, "AgroEye Live Feed Ready",
            (width // 5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2
        )

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
            frame = self._latest_frame if self._latest_frame is not None else self._placeholder

        # Convert OpenCV BGR numpy array to PyAV VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
