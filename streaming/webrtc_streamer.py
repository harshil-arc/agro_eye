import asyncio
import threading
import time
from typing import Optional, Dict, Any, List, Set
import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCIceCandidate
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
    AIORTC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCIceCandidate = None
    candidate_from_sdp = None
    candidate_to_sdp = None
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
    High-performance, Multi-Session Resilient WebRTC Live Streamer for Raspberry Pi 5 (AgroEye).
    Enables mobile app clients to repeatedly connect, disconnect, and reconnect without leaking
    resources or requiring server/camera restarts.
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

        # Session & Connection State
        self.pc: Optional[Any] = None
        self.current_track: Optional[OpenCVVideoTrack] = None
        self.current_session_id: Optional[str] = None
        self.last_offer_timestamp: int = 0
        self._processed_candidates: Set[str] = set()
        self._pending_ice_candidates: List[Any] = []
        self._remote_description_set: bool = False

        # Frame Buffer
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Asyncio Loop & Daemon Thread
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.is_streaming = False

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
        with self._frame_lock:
            self._latest_frame = frame
        self.video_track.update_frame(frame)
        if self.current_track is not None:
            self.current_track.update_frame(frame)

    async def _close_peer_connection(self, target_pc: Optional[Any] = None):
        """
        Completely closes and discards RTCPeerConnection and releases track resources.
        Guarantees clean state for immediate subsequent reconnection.
        """
        pc_to_close = target_pc if target_pc is not None else self.pc
        if pc_to_close is None:
            return

        is_current = (target_pc is None or target_pc == self.pc)
        if is_current:
            self.pc = None
            self._remote_description_set = False
            self._pending_ice_candidates.clear()
            self._processed_candidates.clear()
            self.is_streaming = False
            if self.current_track is not None:
                try:
                    self.current_track.stop()
                except Exception:
                    pass
                self.current_track = None

        try:
            if hasattr(pc_to_close, "getTransceivers"):
                for transceiver in pc_to_close.getTransceivers():
                    try:
                        if transceiver.sender and transceiver.sender.track:
                            transceiver.sender.track.stop()
                    except Exception:
                        pass
            await pc_to_close.close()
            logger.info("🔒 WebRTC: Old RTCPeerConnection closed and all session resources cleared.")
        except Exception as e:
            logger.debug(f"WebRTC cleanup notice: {e}")

    async def _flush_pending_ice_candidates(self):
        """Flushes any ICE candidates received before remoteDescription was set."""
        if not self.pc or not self._remote_description_set or self.pc.remoteDescription is None:
            return

        while self._pending_ice_candidates:
            cand = self._pending_ice_candidates.pop(0)
            try:
                await self.pc.addIceCandidate(cand)
                logger.info("🧊 WebRTC: Flushed queued client ICE candidate successfully.")
            except Exception as e:
                logger.debug(f"Error adding queued client ICE candidate: {e}")

    async def _handle_offer(self, offer_dict: Dict[str, Any]):
        """Processes incoming SDP offer from the Farmer's App and returns an SDP answer."""
        if not self.running:
            return

        session_id = str(offer_dict.get("session_id") or offer_dict.get("sessionId") or offer_dict.get("timestamp") or time.time())
        offer_time = int(offer_dict.get("timestamp", 0) or 0)

        # Reject duplicate offer if already actively streaming for the same session
        if self.pc is not None and self.current_session_id == session_id and self.is_streaming:
            return

        if offer_time > 0 and offer_time <= self.last_offer_timestamp and self.current_session_id == session_id:
            return

        logger.info(f"📡 WebRTC: Received Stream Connection Request (Session: [{session_id}] | TS: {offer_time})")

        # 1. Completely tear down and close any previous RTCPeerConnection
        if self.pc is not None:
            await self._close_peer_connection()

        if not self.running:
            return

        # 2. Reset session tracking
        self.current_session_id = session_id
        self.last_offer_timestamp = offer_time if offer_time > 0 else int(time.time() * 1000)
        self._processed_candidates.clear()
        self._pending_ice_candidates.clear()
        self._remote_description_set = False

        # Clear stale candidates in cloud signaling
        self.signaling.clear_candidates()

        # 3. Create fresh RTCPeerConnection with STUN / TURN configuration
        config = self._build_rtc_config()
        self.pc = RTCPeerConnection(configuration=config)
        current_pc = self.pc

        # 4. Attach fresh video track seeded with latest frame
        self.current_track = OpenCVVideoTrack(fps=self.fps, width=self.width, height=self.height)
        with self._frame_lock:
            if self._latest_frame is not None:
                self.current_track.update_frame(self._latest_frame)

        transceiver = self.pc.addTrack(self.current_track)
        try:
            from aiortc import RTCRtpSender
            caps = RTCRtpSender.getCapabilities("video")
            if caps and caps.codecs:
                # Prioritize H264 / VP8 for mobile playback
                h264_vp8 = [c for c in caps.codecs if c.name.upper() in ("H264", "VP8")]
                if h264_vp8:
                    transceiver.setCodecPreferences(h264_vp8)
        except Exception:
            pass

        # 5. Event Listeners for connection lifecycle
        @current_pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate and self.running and self.pc == current_pc:
                cand_str = f"candidate:{candidate.candidate}" if not str(candidate.candidate).startswith("candidate:") else str(candidate.candidate)
                cand_dict = {
                    "candidate": cand_str,
                    "sdpMid": str(candidate.sdpMid) if candidate.sdpMid is not None else "0",
                    "sdpMLineIndex": int(candidate.sdpMLineIndex) if candidate.sdpMLineIndex is not None else 0,
                    "session_id": session_id,
                    "sessionId": session_id
                }
                self.signaling.send_ice_candidate(cand_dict, session_id=session_id)

        @current_pc.on("connectionstatechange")
        async def on_connection_state():
            if self.pc != current_pc:
                return
            state = current_pc.connectionState
            logger.info(f"🌐 WebRTC Connection State [{session_id}] -> {state.upper()}")
            if state == "connected":
                self.is_streaming = True
                self.signaling.update_status("streaming", {
                    "session_id": session_id,
                    "fps": self.fps,
                    "resolution": f"{self.width}x{self.height}"
                })
            elif state in ("failed", "closed"):
                logger.info(f"WebRTC connection ended ({state}) for session [{session_id}]. Cleaning session.")
                self.is_streaming = False
                self.signaling.update_status("idle")
                await self._close_peer_connection(target_pc=current_pc)

        @current_pc.on("iceconnectionstatechange")
        async def on_ice_state():
            if self.pc != current_pc:
                return
            state = current_pc.iceConnectionState
            logger.info(f"🧊 WebRTC ICE State [{session_id}] -> {state.upper()}")
            if state in ("failed", "closed"):
                logger.warning(f"WebRTC ICE state [{state}] for session [{session_id}]. Resetting session.")
                self.is_streaming = False
                self.signaling.update_status("idle")
                await self._close_peer_connection(target_pc=current_pc)

        # 6. Set Remote Description (Offer)
        try:
            offer = RTCSessionDescription(sdp=offer_dict["sdp"], type=offer_dict.get("type", "offer"))
            await self.pc.setRemoteDescription(offer)
            self._remote_description_set = True
            logger.info(f"✅ WebRTC: Remote description set successfully for session [{session_id}].")
        except Exception as e:
            logger.error(f"Failed to set remote description for session [{session_id}]: {e}")
            await self._close_peer_connection(target_pc=current_pc)
            return

        # 7. Flush queued ICE candidates
        await self._flush_pending_ice_candidates()

        # 8. Create Local Description (Answer) and await ICE gathering
        try:
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # Wait for local STUN/TURN ICE candidates to gather into localDescription (up to 300ms)
            for _ in range(15):
                if current_pc.iceGatheringState == "complete":
                    break
                await asyncio.sleep(0.02)

            # Send complete SDP answer containing gathered host & STUN/TURN candidates
            self.signaling.send_answer(self.pc.localDescription.sdp, self.pc.localDescription.type, session_id=session_id)
            logger.info(f"📡 WebRTC: SDP Answer dispatched to Firebase Signaling Channel for session [{session_id}].")

            # Also publish any individual local ICE candidates to Firebase for trickle ICE compatibility
            for sender in self.pc.getSenders():
                if sender.transport and hasattr(sender.transport, "transport") and hasattr(sender.transport.transport, "iceGatherer"):
                    gatherer = sender.transport.transport.iceGatherer
                    for cand in gatherer.getLocalCandidates():
                        try:
                            cand_clean = candidate_to_sdp(cand) if hasattr(cand, "foundation") else str(cand)
                            cand_str = f"candidate:{cand_clean}" if not str(cand_clean).startswith("candidate:") else str(cand_clean)
                            cand_dict = {
                                "candidate": cand_str,
                                "sdpMid": "0",
                                "sdpMLineIndex": 0,
                                "session_id": session_id,
                                "sessionId": session_id
                            }
                            self.signaling.send_ice_candidate(cand_dict, session_id=session_id)
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Failed to create/set local description for session [{session_id}]: {e}")
            await self._close_peer_connection(target_pc=current_pc)

    async def _poll_signaling_loop(self):
        """Continuous background async loop polling Firebase for incoming stream requests & candidates."""
        logger.info("WebRTC Signaling Watchdog started.")
        self.signaling.update_status("idle", {"device": WEBRTC_DEVICE_ID, "ready": True})

        while self.running:
            try:
                # 0. Check client status for explicit disconnects
                client_status = self.signaling.get_client_status()
                if client_status and isinstance(client_status, dict):
                    status_val = client_status.get("status")
                    status_sid = str(client_status.get("session_id") or client_status.get("sessionId") or "")
                    if status_val == "disconnected" and status_sid and status_sid == self.current_session_id:
                        if self.pc is not None or self.is_streaming:
                            logger.info(f"Client explicitly disconnected session [{status_sid}]. Cleaning peer connection.")
                            await self._close_peer_connection()
                            self.signaling.update_status("idle")

                # 1. Check for incoming stream offers
                offer_dict = self.signaling.get_offer()
                if offer_dict and offer_dict.get("sdp"):
                    offer_sid = str(offer_dict.get("session_id") or offer_dict.get("sessionId") or "")
                    offer_ts = int(offer_dict.get("timestamp", 0) or 0)
                    is_new_session = (offer_sid and offer_sid != self.current_session_id)
                    is_new_ts = (offer_ts > self.last_offer_timestamp)
                    is_not_connected = (self.pc is None or not self.is_streaming)

                    if is_new_session or is_new_ts or (is_not_connected and offer_sid and offer_sid != self.current_session_id):
                        await self._handle_offer(offer_dict)

                # 2. Check for client ICE candidates if connection is forming
                if self.pc and self.pc.connectionState not in ("closed", "failed"):
                    client_candidates = self.signaling.get_client_ice_candidates()
                    for cand in client_candidates:
                        cand_sid = cand.get("session_id") or cand.get("sessionId")
                        # Reject stale candidate from mismatched session
                        if cand_sid and self.current_session_id and cand_sid != self.current_session_id:
                            continue

                        cand_str = str(cand.get("candidate", "")).strip()
                        if cand_str and cand_str not in self._processed_candidates:
                            self._processed_candidates.add(cand_str)
                            try:
                                cand_clean = cand_str[10:].strip() if cand_str.startswith("candidate:") else cand_str
                                ice_cand = candidate_from_sdp(cand_clean)
                                ice_cand.sdpMid = str(cand.get("sdpMid", "0")) if cand.get("sdpMid") is not None else "0"
                                ice_cand.sdpMLineIndex = int(cand.get("sdpMLineIndex", 0)) if cand.get("sdpMLineIndex") is not None else 0

                                if self._remote_description_set and self.pc and self.pc.remoteDescription is not None:
                                    await self.pc.addIceCandidate(ice_cand)
                                else:
                                    self._pending_ice_candidates.append(ice_cand)
                            except Exception as e:
                                logger.debug(f"Error parsing client ICE candidate: {e}")

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
                    asyncio.run_coroutine_threadsafe(self._close_peer_connection(), self.loop)
            except Exception:
                pass
        self.signaling.update_status("offline")
        logger.info("WebRTC Live Streamer stopped.")
