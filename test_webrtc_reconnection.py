import unittest
import asyncio
import numpy as np
import time
from unittest.mock import MagicMock

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
    from aiortc.contrib.media import MediaBlackhole
    AIORTC_AVAILABLE = True
except Exception:
    AIORTC_AVAILABLE = False

from streaming.webrtc_streamer import WebRTCStreamer
from streaming.video_track import OpenCVVideoTrack

class MockSignaling:
    def __init__(self):
        self.offer = None
        self.answer = None
        self.client_candidates = []
        self.pi_candidates = []
        self.status = "idle"

    def get_offer(self):
        return self.offer

    def send_answer(self, sdp, sdp_type="answer", session_id=None):
        self.answer = {
            "sdp": sdp,
            "type": sdp_type,
            "session_id": session_id,
            "timestamp": int(time.time() * 1000)
        }
        return True

    def send_ice_candidate(self, candidate_dict, session_id=None):
        self.pi_candidates.append(candidate_dict)
        return True

    def get_client_ice_candidates(self):
        return list(self.client_candidates)

    def update_status(self, status, extra=None):
        self.status = status

    def clear_candidates(self):
        self.client_candidates.clear()
        self.pi_candidates.clear()

    def clear_session(self):
        self.offer = None
        self.answer = None
        self.client_candidates.clear()
        self.pi_candidates.clear()
        self.status = "idle"

class TestWebRTCReconnection(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(AIORTC_AVAILABLE, "aiortc is required for WebRTC tests")
    async def test_repeated_reconnections_10_times(self):
        """
        Simulates an Android client connecting, streaming, disconnecting, and reconnecting
        for 10 full cycles without restarting the WebRTC streamer or camera pipeline.
        """
        print("\n" + "="*70)
        print(" STARTING 10-CYCLE REPEATED WEBRTC RECONNECTION TEST (ANDROID SIMULATION)")
        print("="*70)

        streamer = WebRTCStreamer()
        mock_sig = MockSignaling()
        streamer.signaling = mock_sig
        streamer.running = True

        # Provide simulated camera frames continuously
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:, :] = (0, 200, 0)
        streamer.update_frame(test_frame)

        for cycle in range(1, 11):
            session_id = f"android_session_{cycle}_{int(time.time()*1000)}"
            print(f"\n--- [Cycle {cycle}/10] Initializing fresh Android WebRTC session: {session_id} ---")

            # 1. Android Client creates its RTCPeerConnection and adds a transceiver to receive video
            client_pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
            client_pc.addTransceiver("video", direction="recvonly")

            # 2. Android Client creates SDP offer
            client_offer = await client_pc.createOffer()
            await client_pc.setLocalDescription(client_offer)

            offer_dict = {
                "sdp": client_pc.localDescription.sdp,
                "type": client_pc.localDescription.type,
                "session_id": session_id,
                "sessionId": session_id,
                "timestamp": int(time.time() * 1000)
            }
            mock_sig.offer = offer_dict

            # 3. Pi WebRTC Streamer handles offer
            await streamer._handle_offer(offer_dict)
            self.assertIsNotNone(streamer.pc, f"Cycle {cycle}: Streamer PC was not created")
            self.assertEqual(streamer.current_session_id, session_id)
            self.assertIsNotNone(mock_sig.answer, f"Cycle {cycle}: SDP Answer was not sent")
            self.assertEqual(mock_sig.answer["session_id"], session_id)

            # 4. Android Client sets remote description from Pi's answer
            answer_desc = RTCSessionDescription(
                sdp=mock_sig.answer["sdp"],
                type=mock_sig.answer["type"]
            )
            await client_pc.setRemoteDescription(answer_desc)

            # 5. Push frames to verify streaming pipeline remains live
            for f in range(3):
                test_frame[10:50, 10:50] = (f * 30, 120, 200)
                streamer.update_frame(test_frame)

            self.assertIsNotNone(streamer.current_track, f"Cycle {cycle}: Video track should be active")

            # 6. Android Client DISCONNECTS and closes session
            print(f"--- [Cycle {cycle}/10] Android Client disconnecting session {session_id} ---")
            await client_pc.close()

            # Trigger server close callback / cleanup
            await streamer._close_peer_connection()

            # Verify server state is clean and ready for immediate next connection
            self.assertIsNone(streamer.pc, f"Cycle {cycle}: Streamer PC should be None after close")
            self.assertIsNone(streamer.current_track, f"Cycle {cycle}: Video track should be None after close")
            self.assertEqual(len(streamer._processed_candidates), 0)
            self.assertEqual(len(streamer._pending_ice_candidates), 0)
            self.assertFalse(streamer.is_streaming)
            print(f"[OK] Cycle {cycle}/10 passed: Session cleanly closed and server ready for immediate reconnection.")

            # Small yield to event loop
            await asyncio.sleep(0.01)

        streamer.stop()
        print("\n" + "="*70)
        print(" ALL 10 CONSECUTIVE RECONNECTION CYCLES SUCCEEDED WITH 0 FAILURES!")
        print("="*70 + "\n")

    @unittest.skipUnless(AIORTC_AVAILABLE, "aiortc is required for WebRTC tests")
    async def test_ice_candidate_queueing_before_remote_description(self):
        """
        Verifies that ICE candidates arriving before remoteDescription are safely queued
        and flushed without crashing or causing InvalidStateErrors.
        """
        streamer = WebRTCStreamer()
        mock_sig = MockSignaling()
        streamer.signaling = mock_sig
        streamer.running = True

        from aiortc.sdp import candidate_from_sdp
        sample_cand = candidate_from_sdp("1 1 UDP 2122252543 192.168.1.100 50000 typ host")
        sample_cand.sdpMid = "0"
        sample_cand.sdpMLineIndex = 0

        # Simulate early candidate arrival before remote description
        streamer._pending_ice_candidates.append(sample_cand)
        self.assertEqual(len(streamer._pending_ice_candidates), 1)

        # Create PC and set remote description
        client_pc = RTCPeerConnection()
        client_pc.addTransceiver("video", direction="recvonly")
        client_offer = await client_pc.createOffer()

        offer_dict = {
            "sdp": client_offer.sdp,
            "type": client_offer.type,
            "session_id": "queue_test_session",
            "timestamp": int(time.time() * 1000)
        }
        await streamer._handle_offer(offer_dict)

        # After _handle_offer, pending queue must be flushed
        self.assertEqual(len(streamer._pending_ice_candidates), 0)
        print("[OK] Early ICE candidate queuing and flushing verified successfully.")

        await client_pc.close()
        await streamer._close_peer_connection()
        streamer.stop()

if __name__ == '__main__':
    unittest.main()
