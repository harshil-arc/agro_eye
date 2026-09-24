import unittest
import numpy as np
import asyncio
from streaming.video_track import OpenCVVideoTrack
from streaming.signaling import FirebaseWebRTCSignaling
from streaming.webrtc_streamer import WebRTCStreamer

class TestWebRTCStreaming(unittest.TestCase):
    def test_video_track_frame_conversion(self):
        track = OpenCVVideoTrack(fps=20, width=640, height=480)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (0, 255, 0)  # Pure green
        track.update_frame(test_frame)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            av_frame = loop.run_until_complete(track.recv())
            self.assertIsNotNone(av_frame)
            self.assertEqual(av_frame.width, 640)
            self.assertEqual(av_frame.height, 480)
            print(" OpenCVVideoTrack frame encoding passed (640x480 av.VideoFrame)")
        finally:
            loop.close()

    def test_signaling_initialization(self):
        signaling = FirebaseWebRTCSignaling()
        self.assertIsNotNone(signaling.session_path)
        self.assertIn("webrtc_sessions", signaling.session_path)
        print(f" FirebaseWebRTCSignaling initialized at path: {signaling.session_path}")

    def test_streamer_initialization(self):
        streamer = WebRTCStreamer()
        self.assertTrue(streamer.enabled)
        self.assertEqual(streamer.width, 640)
        self.assertEqual(streamer.height, 480)
        config = streamer._build_rtc_config()
        self.assertGreater(len(config.iceServers), 0)
        print(f" WebRTCStreamer initialized with {len(config.iceServers)} ICE servers")

if __name__ == '__main__':
    unittest.main()
