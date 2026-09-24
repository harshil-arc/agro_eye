import time
import fractions
import asyncio
import cv2
import numpy as np
import av
from aiortc import VideoStreamTrack
from typing import Optional
from utils.logger import logger

class OpenCVVideoTrack(VideoStreamTrack):
    """
    Custom WebRTC Video Stream Track that sources frames from AgroEye's live OpenCV camera buffer.
    Ensures precise frame rate timing, color space conversion (BGR -> av.VideoFrame), and
    graceful fallback if a frame is momentarily missed.
    """
    kind = "video"

    def __init__(self, fps: int = 20, width: int = 640, height: int = 480):
        super().__init__()
        self.fps = fps
        self.width = width
        self.height = height
        self._start: Optional[float] = time.time()
        self._timestamp = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        self._last_frame_time = time.time()
        
        # Placeholder standby frame if camera starts before frames arrive
        self._placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            self._placeholder, "AgroEye Camera Initializing...",
            (width // 6, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2
        )

    def update_frame(self, frame: np.ndarray):
        """Thread-safe update of the latest captured/annotated video frame."""
        if frame is None:
            return
        
        # Resize if dimension differs from streaming target
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame_resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        self._latest_frame = frame_resized
        self._last_frame_time = time.time()

    async def recv(self) -> av.VideoFrame:
        """
        WebRTC engine calls this method periodically to pull the next video frame.
        """
        pts, time_base = await self.next_timestamp()

        # Target frame interval
        frame_interval = 1.0 / self.fps
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed < frame_interval:
            await asyncio.sleep(frame_interval - elapsed)

        frame = self._latest_frame if self._latest_frame is not None else self._placeholder

        # Convert OpenCV BGR numpy array to PyAV VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
