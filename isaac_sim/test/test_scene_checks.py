import ast
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scene_checks

ISAAC_DIR = os.path.join(os.path.dirname(__file__), "..")


def _passing_report():
    return {
        "gravity_configured_m_s2": 3.72,
        "gravity_measured_m_s2": 3.70,
        "settled": True,
        "camera": {"mean": 90.0, "std": 20.0},
        "drive_travel_m": 0.7,
    }


class TestSpawnHeight:
    def test_clears_highest_point_in_footprint(self):
        elevation = np.zeros((101, 101))
        elevation[50, 52] = 7.0
        assert scene_checks.spawn_height(elevation, 100.0, 100.0) == 8.0

    def test_ignores_terrain_outside_footprint(self):
        elevation = np.zeros((101, 101))
        elevation[0, 0] = 50.0
        assert scene_checks.spawn_height(elevation, 100.0, 100.0) == 1.0


class TestGravityFromDrop:
    @pytest.mark.parametrize("g", [3.72, 9.81])
    def test_recovers_gravity_from_semi_implicit_integration(self, g):
        dt, velocity, z = 1.0 / 60.0, 0.0, 0.0
        for _ in range(12):
            velocity += g * dt
            z += velocity * dt
        assert scene_checks.gravity_from_drop(z, 12, dt) == pytest.approx(g)

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            scene_checks.gravity_from_drop(1.0, 0, 0.1)


class TestImageStats:
    def test_ignores_alpha(self):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        assert scene_checks.image_stats(rgba) == {"mean": 0.0, "std": 0.0}

    def test_empty_image_counts_as_black(self):
        assert scene_checks.image_stats(np.array([])) == {"mean": 0.0, "std": 0.0}


class TestEvaluateSmoke:
    def test_good_report_passes(self):
        assert scene_checks.evaluate_smoke(_passing_report()) == []

    @pytest.mark.parametrize("change", [
        {"gravity_configured_m_s2": 9.81},
        {"gravity_measured_m_s2": 9.81},
        {"settled": False},
        {"camera": {"mean": 0.0, "std": 0.0}},
        {"camera": {"mean": 120.0, "std": 0.0}},
        {"drive_travel_m": 0.0},
    ])
    def test_each_failure_is_caught(self, change):
        assert len(scene_checks.evaluate_smoke({**_passing_report(), **change})) == 1

    def test_empty_report_fails(self):
        assert scene_checks.evaluate_smoke({})


class TestIsaacScriptsStayIndependentOfSystemRos:
    @pytest.mark.parametrize("name", ["setup_scene.py", "rover_controller.py"])
    def test_no_rclpy_or_old_isaac_imports(self, name):
        with open(os.path.join(ISAAC_DIR, name), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not [m for m in imported if m.split(".")[0] == "rclpy" or m.startswith("omni.isaac")]


class TestRoverController:
    def _controller(self, command):
        from rover_controller import RoverWheelController
        return RoverWheelController(None, {"root_path": "/World/Rover"}, read_command=lambda: command)

    @pytest.fixture(autouse=True)
    def _no_usd(self, monkeypatch):
        import rover_controller
        monkeypatch.setattr(rover_controller, "_PXR_AVAILABLE", False)

    def test_step_uses_read_command(self):
        controller = self._controller((0.1, 0.0))
        controller.step()
        assert controller._cmd_linear == 0.1

    def test_override_wins_over_read_command(self):
        controller = self._controller((0.1, 0.0))
        controller.set_override(0.3, 0.0)
        controller.step()
        assert controller._cmd_linear == 0.3

    def test_stop_stays_stopped(self):
        controller = self._controller((0.1, 0.0))
        controller.stop()
        controller.step()
        assert controller._cmd_linear == 0.0
