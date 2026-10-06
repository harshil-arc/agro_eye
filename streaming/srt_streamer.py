import os
import sys
import time
import shutil
import threading
import subprocess
from typing import Optional
import numpy as np

from config import (
    ENABLE_STREAMING, STREAM_FPS, STREAM_WIDTH, STREAM_HEIGHT,
    STREAM_BITRATE, SRT_SERVER_URL, SRT_STREAM_ID,
    SRT_LATENCY_MS, SRT_RECONNECT_DELAY,
    STREAM_DEVICE_ID, HLS_STREAM_URL
)
from utils.logger import logger
from firebase.realtime_db import RealtimeDatabaseManager

class SRTStreamer:
    """
    High-Performance Outbound SRT Live Video Streamer for Raspberry Pi 5 (AgroEye).
    Directly encodes raw BGR frames to low-latency H.264 via FFmpeg and streams
    outbound over SRT in caller mode to a remote Cloud VPS or Media Server.
    
    Features:
    - 0% Port Forwarding required (Pure outbound UDP caller connection).
    - 1280x720 @ 30 FPS hardware/ultrafast H.264 single-pass encoding.
    - Zero-copy single-frame buffer preventing frame lag or memory bloat.
    - Automatic reconnection watchdog on network disconnect or server downtime.
    - Real-time /live_stream metadata synchronization to Firebase RTDB.
    - Non-blocking: Farm AI detection and sensor loops continue running uninterrupted.
    """
    def __init__(
        self,
        server_url: Optional[str] = None,
        stream_id: Optional[str] = None,
        hls_url: Optional[str] = None,
        device_id: Optional[str] = None,
        width: int = STREAM_WIDTH,
        height: int = STREAM_HEIGHT,
        fps: int = STREAM_FPS,
        bitrate: str = STREAM_BITRATE,
        latency_ms: int = SRT_LATENCY_MS,
        reconnect_delay: float = SRT_RECONNECT_DELAY
    ):
        self.enabled = ENABLE_STREAMING
        self.server_url = server_url or SRT_SERVER_URL
        self.stream_id = stream_id or SRT_STREAM_ID
        self.hls_url = hls_url or HLS_STREAM_URL
        self.device_id = device_id or STREAM_DEVICE_ID
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.latency_ms = latency_ms
        self.reconnect_delay = reconnect_delay

        self.rtdb = RealtimeDatabaseManager()
        self.running = False
        self.is_connected = False
        self.process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        # Thread-safe single-frame buffer
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()
        self._ffmpeg_bin = shutil.which("ffmpeg")

        if self.enabled and not self._ffmpeg_bin:
            logger.warning("FFmpeg not found in system PATH. Please install FFmpeg (sudo apt install -y ffmpeg) for SRT live video streaming.")

    def get_full_srt_url(self) -> str:
        """Constructs full SRT output URL with caller mode, stream ID, and low-latency tuning."""
        base = self.server_url.strip()
        if not base.startswith("srt://"):
            base = f"srt://{base}"

        parts = base.split("?", 1)
        url_base = parts[0]
        query_str = parts[1] if len(parts) > 1 else ""

        params = {}
        if query_str:
            for item in query_str.split("&"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    params[k] = v

        # Caller mode initiates outgoing connection to VPS
        params.setdefault("mode", "caller")
        # Latency in microseconds for FFmpeg libsrt (e.g. 120ms = 120000us)
        params.setdefault("latency", str(int(self.latency_ms * 1000)))
        params.setdefault("pkt_size", "1316")

        if self.stream_id:
            params.setdefault("streamid", self.stream_id)

        query_parts = [f"{k}={v}" for k, v in params.items()]
        return f"{url_base}?{'&'.join(query_parts)}"

    def update_frame(self, frame: np.ndarray):
        """
        Called from main camera loop to deliver the latest frame.
        Instantaneous and non-blocking (<0.05ms). Never locks AI or sensor pipelines.
        """
        if not self.enabled or frame is None or not self.running:
            return

        with self._lock:
            self._latest_frame = frame
            self._new_frame_event.set()

    def _build_ffmpeg_cmd(self, srt_target_url: str) -> list:
        """Builds optimized low-latency single-pass H.264 FFmpeg command."""
        return [
            self._ffmpeg_bin,
            "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps),
            "-keyint_min", str(max(5, self.fps // 2)),
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", "500k",
            "-flush_packets", "1",
            "-f", "mpegts",
            srt_target_url
        ]

    def _cleanup_process(self):
        """Safely closes FFmpeg standard input pipe and terminates process."""
        self.is_connected = False
        if self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception:
                pass
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def _stream_worker(self):
        """
        Background worker maintaining outgoing SRT stream connection and piping frames.
        Automatically reconnects if the internet or remote media server goes offline.
        """
        target_url = self.get_full_srt_url()

        while self.running:
            if not self._ffmpeg_bin:
                time.sleep(5.0)
                continue

            try:
                # 1. Initiate SRT Connection
                logger.info(f"📡 SRT connecting to {self.server_url} (Stream ID: {self.stream_id})...")
                cmd = self._build_ffmpeg_cmd(target_url)
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=10**7
                )

                frames_sent = 0
                first_frame_logged = False

                # 2. Feed Frames to FFmpeg
                while self.running and self.process.poll() is None:
                    # Wait for a new frame from camera
                    signaled = self._new_frame_event.wait(timeout=0.1)
                    if not signaled:
                        continue

                    self._new_frame_event.clear()
                    with self._lock:
                        frame = self._latest_frame

                    if frame is None:
                        continue

                    # Resize frame if dimensions don't match target stream
                    h, w = frame.shape[:2]
                    if w != self.width or h != self.height:
                        import cv2
                        frame_to_send = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_to_send = frame

                    try:
                        self.process.stdin.write(frame_to_send.tobytes())
                        self.process.stdin.flush()
                        frames_sent += 1

                        if not first_frame_logged and frames_sent >= 2:
                            self.is_connected = True
                            first_frame_logged = True
                            logger.info(f"🌐 SRT connected (Publishing live stream -> {self.server_url} | ID: {self.stream_id} @ {self.width}x{self.height} {self.fps}FPS)")
                            # Publish "live" stream metadata to Firebase RTDB for instant mobile app playback
                            self._publish_metadata(status="live")

                    except (BrokenPipeError, IOError):
                        logger.warning(f"⚠️ SRT disconnected (Connection to {self.server_url} lost)")
                        break

                # 3. Connection closed or terminated
                if self.is_connected:
                    logger.warning(f"⚠️ SRT disconnected (Stream connection closed)")
                    self._publish_metadata(status="standby")

                stderr_out = ""
                if self.process and self.process.stderr:
                    try:
                        stderr_out = self.process.stderr.read().decode("utf-8", errors="replace").strip()
                    except Exception:
                        pass

                if stderr_out and "Connection refused" not in stderr_out:
                    logger.debug(f"FFmpeg notice: {stderr_out}")

            except Exception as e:
                logger.error(f"❌ Stream error: {e}")

            finally:
                self._cleanup_process()

            if self.running:
                logger.info(f"🔄 Reconnecting to SRT server in {self.reconnect_delay:.0f}s...")
                # Interruptible sleep
                sleep_start = time.time()
                while self.running and (time.time() - sleep_start) < self.reconnect_delay:
                    time.sleep(0.2)

    def _publish_metadata(self, status: str = "live"):
        """Publishes current stream URL and status to Firebase /live_stream/<device_id>."""
        try:
            payload = {
                "stream_url": self.hls_url,
                "stream_status": status,
                "fps": self.fps,
                "resolution": f"{self.width}x{self.height}",
                "last_active": int(time.time() * 1000)
            }
            self.rtdb.update_live_stream_metadata(self.device_id, payload)
        except Exception as e:
            logger.debug(f"Error publishing stream metadata to Firebase: {e}")

    def start(self):
        """Starts the outbound SRT streaming engine in a background daemon thread."""
        if not self.enabled:
            logger.info("SRT Live Video Streaming is disabled in configuration.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._stream_worker, daemon=True, name="SRTStreamer")
        self._thread.start()
        logger.info(f"SRT Live Streamer active (Destination: {self.server_url} | Target: {self.width}x{self.height} @ {self.fps}FPS)")

    def stop(self):
        """Gracefully stops the SRT streamer and releases FFmpeg process resources."""
        if not self.running:
            return
        self.running = False
        self._publish_metadata(status="offline")
        self._new_frame_event.set()
        self._cleanup_process()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
        logger.info("SRT Live Streamer stopped.")
