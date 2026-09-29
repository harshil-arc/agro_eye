import unittest
import time
from servo.servo_controller import ServoController, CameraControlState

class TestServoController(unittest.TestCase):
    def setUp(self):
        self.controller = ServoController(
            pan_pin=18,
            tilt_pin=13,
            min_angle=30,
            max_angle=150,
            step_deg=5,
            step_interval=0.01,
            pause_sec=0.01
        )

    def tearDown(self):
        self.controller.stop()

    def test_initial_state(self):
        self.assertEqual(self.controller.mode, "auto")
        self.assertEqual(self.controller.current_pan, 90.0)
        self.assertEqual(self.controller.current_tilt, 90.0)

    def test_angle_to_duty_cycle(self):
        # 0 deg => 2.5% duty
        # 90 deg => 7.5% duty
        # 180 deg => 12.5% duty
        self.assertAlmostEqual(self.controller._angle_to_duty(0), 2.5, places=1)
        self.assertAlmostEqual(self.controller._angle_to_duty(90), 7.5, places=1)
        self.assertAlmostEqual(self.controller._angle_to_duty(180), 12.5, places=1)

    def test_manual_mode_and_position(self):
        self.controller.set_position(45.0, 90.0, command="set_coords")
        self.assertEqual(self.controller.mode, "manual")
        self.assertEqual(self.controller.current_pan, 45.0)
        self.assertEqual(self.controller.target_pan, 45.0)

    def test_mode_toggle(self):
        self.controller.set_mode("manual")
        self.assertEqual(self.controller.mode, "manual")
        self.controller.set_mode("auto")
        self.assertEqual(self.controller.mode, "auto")

    def test_camera_control_state_dict(self):
        state = CameraControlState(
            mode="manual",
            pan_angle=120,
            tilt_angle=90,
            command="pan_right"
        )
        d = state.to_dict()
        self.assertEqual(d["mode"], "manual")
        self.assertEqual(d["pan_angle"], 120)
        self.assertEqual(d["tilt_angle"], 90)
        self.assertEqual(d["x_coord"], 30)  # 120 - 90
        self.assertEqual(d["y_coord"], 0)   # 90 - 90
        self.assertEqual(d["command"], "pan_right")

    def test_lifecycle_start_stop(self):
        self.controller.start()
        self.assertTrue(self.controller.running)
        time.sleep(0.1)
        self.controller.stop()
        self.assertFalse(self.controller.running)

if __name__ == "__main__":
    unittest.main()
