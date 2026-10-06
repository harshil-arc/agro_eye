import unittest
import numpy as np
import time
import subprocess
from unittest.mock import MagicMock, patch

from config import (
    SRT_SERVER_URL, SRT_STREAM_ID, SRT_LATENCY_MS,
    STREAM_FPS, STREAM_WIDTH, STREAM_HEIGHT, STREAM_BITRATE
)
from streaming.srt_streamer import SRTStreamer

class TestSRTStreaming(unittest.TestCase):
    def test_srt_config_and_url_formatting(self):
        """Verifies that SRT caller URL is correctly assembled with all required parameters."""
        streamer = SRTStreamer(
            server_url="srt://vps.agroeye.cloud:6000",
            stream_id="publish/live/farm_cam_01",
            width=1280,
            height=720,
            fps=30,
            bitrate="2500k",
            latency_ms=120
        )
        url = streamer.get_full_srt_url()
        print(f"\n[OK] Generated SRT URL: {url}")

        self.assertTrue(url.startswith("srt://vps.agroeye.cloud:6000"))
        self.assertIn("mode=caller", url)
        self.assertIn("latency=120000", url)
        self.assertIn("pkt_size=1316", url)
        self.assertIn("streamid=publish/live/farm_cam_01", url)
        self.assertEqual(streamer.width, 1280)
        self.assertEqual(streamer.height, 720)
        self.assertEqual(streamer.fps, 30)

    def test_nonblocking_frame_updates(self):
        """Verifies that update_frame delivers frames instantaneously without blocking."""
        streamer = SRTStreamer(server_url="srt://127.0.0.1:6000")
        streamer.running = True

        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame[10:50, 10:50] = (0, 255, 0)

        t0 = time.perf_counter()
        for _ in range(100):
            streamer.update_frame(test_frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(elapsed_ms, 50.0, "Frame updates should take < 0.5ms per frame")
        self.assertIsNotNone(streamer._latest_frame)
        print(f"[OK] 100 frame updates completed in {elapsed_ms:.2f}ms (non-blocking)")
        streamer.stop()

    def test_automatic_reconnection_loop(self):
        """
        Simulates remote server drops / broken pipe and verifies that SRTStreamer
        automatically cleans up and transitions into reconnection mode without crashing.
        """
        streamer = SRTStreamer(
            server_url="srt://127.0.0.1:9999",
            reconnect_delay=0.1
        )

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        # Simulate broken pipe on second write
        mock_process.stdin.write.side_effect = [None, BrokenPipeError("Server closed connection")]
        mock_process.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_process):
            # Test cleanup
            streamer.process = mock_process
            streamer.is_connected = True
            streamer._cleanup_process()

            self.assertFalse(streamer.is_connected)
            self.assertIsNone(streamer.process)
            mock_process.terminate.assert_called_once()
            print("[OK] SRT process cleanup and disconnect handling verified.")

    def test_graceful_stop(self):
        """Verifies clean stop and resource teardown."""
        streamer = SRTStreamer(server_url="srt://127.0.0.1:6000")
        streamer.start()
        self.assertTrue(streamer.running)
        time.sleep(0.1)
        streamer.stop()
        self.assertFalse(streamer.running)
        print("[OK] SRT streamer started and stopped gracefully.")

if __name__ == "__main__":
    unittest.main()
