import asyncio
import threading
import time
from typing import Optional, Dict, Any
import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCIceCandidate
    from aiortc.sdp import candidate_from_sdp
    AIORTC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCIceCandidate = None
    candidate_from_sdp = None
    AIORTC_AVAILABLE = False

from config import (
    ENABLE_STREAMING, STREAM_FPS, STREAM_WIDTH, STREAM_HEIGHT,
    STUN_SERVERS, TURN_SERVERS_CONFIG,
    WEBRTC_DEVICE_ID
)
from utils.logger import logger
from streaming.video_track import OpenCVVideoTrack
from streaming.signaling import FirebaseWebRTCSignaling

class WebRTCStreamer:
    """
    High-performance WebRTC Live Streamer for Raspberry Pi (AgroEye).
    Enables farmers to view ultra-low latency real-time farm camera feed over the Internet
    with automatic STUN/TURN NAT traversal.
    """
    def __init__(self):
        self.enabled = ENABLE_STREAMING and AIORTC_AVAILABLE
        if ENABLE_STREAMING and not AIORTC_AVAILABLE:
            logger.warning("WebRTC Live Streaming disabled: 'aiortc' package not found in this Python environment.")

        self.fps = STREAM_FPS
        self.width = STREAM_WIDTH
        self.height = STREAM_HEIGHT
        
        self.video_track = OpenCVVideoTrack(fps=self.fps, width=self.width, height=self.height)
        self.signaling = FirebaseWebRTCSignaling()
        
        self.pc: Optional[Any] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.is_streaming = False
        self.last_offer_timestamp = 0
        self._processed_candidates = set()

    def _build_rtc_config(self) -> RTCConfiguration:
        """Constructs RTCConfiguration with Google STUN + Multi-Port TURN relay servers."""
        ice_servers = []
        for stun in STUN_SERVERS:
            ice_servers.append(RTCIceServer(urls=stun))

        for turn_cfg in TURN_SERVERS_CONFIG:
            ice_servers.append(RTCIceServer(
                urls=turn_cfg["urls"],
                username=turn_cfg.get("username"),
                credential=turn_cfg.get("credential")
            ))

        return RTCConfiguration(iceServers=ice_servers)

    def update_frame(self, frame: np.ndarray):
        """Called from main camera loop to provide latest frame for streaming."""
        if not self.enabled or frame is None:
            return
        self.video_track.update_frame(frame)

    async def _handle_offer(self, offer_dict: Dict[str, Any]):
        """Processes incoming SDP offer from the Farmer's App and returns an SDP answer."""
        if not self.running:
            return

        offer_time = offer_dict.get("timestamp", 0)
        if offer_time <= self.last_offer_timestamp:
            return

        logger.info(f"📡 WebRTC: Received Stream Connection Request from Farmer's App (TS: {offer_time})")
        self.last_offer_timestamp = offer_time
        self._processed_candidates.clear()

        # Clean up any prior peer connection
        if self.pc is not None:
            try:
                await self.pc.close()
            except Exception:
                pass
            self.pc = None

        if not self.running:
            return

        # Create new RTCPeerConnection with STUN / TURN configuration
        config = self._build_rtc_config()
        self.pc = RTCPeerConnection(configuration=config)

        # Attach custom video track
        transceiver = self.pc.addTrack(self.video_track)
        try:
            from aiortc import RTCRtpSender
            caps = RTCRtpSender.getCapabilities("video")
            if caps and caps.codecs:
                # Prioritize H264 / VP8 for hardware & low-latency mobile playback
                h264_vp8 = [c for c in caps.codecs if c.name.upper() in ("H264", "VP8")]
                if h264_vp8:
                    transceiver.setCodecPreferences(h264_vp8)
        except Exception:
            pass

        @self.pc.on("connectionstatechange")
        async def on_connection_state():
            if self.pc is None:
                return
            state = self.pc.connectionState
            logger.info(f"🌐 WebRTC Connection State -> {state.upper()}")
            if state == "connected":
                self.is_streaming = True
                self.signaling.update_status("streaming", {
                    "fps": self.fps,
                    "resolution": f"{self.width}x{self.height}"
                })
            elif state in ("failed", "closed", "disconnected"):
                self.is_streaming = False
                self.signaling.update_status("idle")
                # Reset timestamp so client can reconnect immediately
                if state == "failed":
                    logger.warning("WebRTC connection failed. Resetting session for next attempt.")
                    self.last_offer_timestamp = 0

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state():
            if self.pc is None:
                return
            state = self.pc.iceConnectionState
            logger.info(f"🧊 WebRTC ICE Connection State -> {state.upper()}")
            if state == "failed":
                logger.warning("WebRTC ICE negotiation failed. Enabling fallback reconnect...")
                self.last_offer_timestamp = 0

        # 1. Set Remote Description (Offer)
        offer = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict.get("type", "offer"))
        await self.pc.setRemoteDescription(offer)

        # 2. Create Local Description (Answer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        # 3. Send Answer to Firebase Signaling Channel
        self.signaling.send_answer(self.pc.localDescription.sdp, self.pc.localDescription.type)
        logger.info("📡 WebRTC: SDP Answer dispatched to Firebase Signaling Channel.")

    async def _poll_signaling_loop(self):
        """Continuous background async loop polling Firebase for incoming stream requests & candidates."""
        logger.info("WebRTC Signaling Watchdog started.")
        self.signaling.update_status("idle", {"device": WEBRTC_DEVICE_ID, "ready": True})

        while self.running:
            try:
                # 1. Check for incoming stream offers
                offer_dict = self.signaling.get_offer()
                if offer_dict and offer_dict.get("sdp"):
                    offer_ts = offer_dict.get("timestamp", 0)
                    if offer_ts > self.last_offer_timestamp:
                        await self._handle_offer(offer_dict)

                # 2. Check for client ICE candidates if connection is forming
                if self.pc and self.pc.connectionState not in ("closed", "failed"):
                    client_candidates = self.signaling.get_client_ice_candidates()
                    for cand in client_candidates:
                        cand_str = cand.get("candidate", "")
                        if cand_str and cand_str not in self._processed_candidates:
                            self._processed_candidates.add(cand_str)
                            try:
                                cand_clean = cand_str.replace("candidate:", "").strip()
                                ice_cand = candidate_from_sdp(cand_clean)
                                ice_cand.sdpMid = str(cand.get("sdpMid", "0")) if cand.get("sdpMid") is not None else "0"
                                ice_cand.sdpMLineIndex = int(cand.get("sdpMLineIndex", 0)) if cand.get("sdpMLineIndex") is not None else 0
                                await self.pc.addIceCandidate(ice_cand)
                            except Exception as e:
                                logger.debug(f"Error adding client ICE candidate: {e}")

            except Exception as e:
                logger.debug(f"WebRTC signaling poll error: {e}")

            # Ultra-fast responsive poll timing:
            if self.pc and self.pc.connectionState in ("connecting", "checking", "new"):
                await asyncio.sleep(0.04)  # 40ms ultra-fast handshake
            elif self.is_streaming:
                await asyncio.sleep(0.4)   # 400ms during active stream
            else:
                await asyncio.sleep(0.08)  # 80ms for instant app connection response


    def _run_event_loop(self):
        """Target for background daemon thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        def _silent_asyncio_handler(loop, context):
            exception = context.get("exception")
            message = context.get("message", "")
            # Catch and silence harmless STUN socket retries / closed transport callbacks
            if "Fatal write error" in message or "sendto" in str(exception) or "NoneType" in str(exception) or "aioice" in str(context):
                return
            logger.debug(f"WebRTC asyncio event: {message} ({exception})")

        self.loop.set_exception_handler(_silent_asyncio_handler)

        try:
            self.loop.run_until_complete(self._poll_signaling_loop())
        except Exception as e:
            logger.debug(f"WebRTC event loop terminated: {e}")
        finally:
            if self.loop.is_running():
                self.loop.close()

    def start(self):
        """Starts the WebRTC streaming engine in a non-blocking background thread."""
        if not self.enabled:
            logger.info("WebRTC Streaming is disabled in configuration.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        logger.info(f" WebRTC Live Streamer active (Device: {WEBRTC_DEVICE_ID} | Target: {self.width}x{self.height} @ {self.fps}FPS)")

    def stop(self):
        """Gracefully shuts down the WebRTC streaming engine."""
        self.running = False
        if self.pc:
            try:
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
            except Exception:
                pass
        self.signaling.update_status("offline")
        logger.info("WebRTC Live Streamer stopped.")
