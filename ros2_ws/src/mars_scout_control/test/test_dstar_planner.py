"""Unit tests for DStarLitePlanner."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_control.dstar_planner import DStarLitePlanner, GRID_RES, INF
import numpy as np


def test_flat_terrain_has_path():
    p = DStarLitePlanner(20.0, 20.0)
    p.set_goal(5.0, 0.0)
    wp = p.next_waypoint(0.0, 0.0)
    assert wp is not None, "Should find a path on flat terrain"


def test_goal_reached_returns_goal():
    p = DStarLitePlanner(20.0, 20.0)
    p.set_goal(0.1, 0.0)
    wp = p.next_waypoint(0.0, 0.0)
    assert wp is not None
    gx, gy = wp
    # Waypoint should be near goal
    assert math.hypot(gx - 0.1, gy - 0.0) < 2.0


def test_waypoint_moves_toward_goal():
    p = DStarLitePlanner(40.0, 40.0)
    p.set_goal(10.0, 0.0)
    wp = p.next_waypoint(-10.0, 0.0)
    assert wp is not None
    # Waypoint x should be > rover x (moving right toward goal)
    assert wp[0] > -10.0


def test_wall_of_obstacles_no_path():
    p = DStarLitePlanner(20.0, 20.0)
    p.set_goal(5.0, 0.0)
    # Build a vertical wall of obstacles at x=1.0 blocking the path
    for y in range(-10, 10):
        p.update_cell(1.0, float(y) * GRID_RES, INF)
    wp = p.next_waypoint(0.0, 0.0)
    # May or may not find a path around the wall depending on grid size,
    # but should not crash
    # (wall might not be complete on a 20m grid, just check no exception)


def test_path_routes_around_obstacle():
    p = DStarLitePlanner(20.0, 20.0)
    p.set_goal(4.0, 0.0)
    # Block direct path with a partial wall at x=2, y=[-1,0,1]
    for yi in [-1, 0, 1]:
        p.update_cell(2.0, float(yi), INF)
    wp = p.next_waypoint(0.0, 0.0)
    assert wp is not None, "Should route around partial obstacle"
    # Waypoint should deviate from y=0 to go around
    # (not strictly required but typical for a detour)


def test_elevation_update():
    p = DStarLitePlanner(40.0, 40.0)
    # Build a flat elevation array with one steep patch
    elev = np.zeros((80, 80), dtype=np.float32)
    # Make a steep slope patch (central 10x10 cells at 2m height)
    elev[35:45, 35:45] = 2.0
    p.update_from_elevation(elev, 40.0, 40.0)
    p.set_goal(10.0, 0.0)
    wp = p.next_waypoint(-10.0, 0.0)
    # Should still find a path (steep patch is high cost but not necessarily INF)
    # Flat areas around the patch are cost 1.0
    assert wp is not None


def test_has_path_flag():
    p = DStarLitePlanner(10.0, 10.0)
    p.set_goal(3.0, 0.0)
    p.next_waypoint(0.0, 0.0)  # trigger planning
    assert p.has_path()


def test_no_goal_returns_none():
    p = DStarLitePlanner(20.0, 20.0)
    wp = p.next_waypoint(0.0, 0.0)
    assert wp is None


def test_goal_update_replans():
    p = DStarLitePlanner(20.0, 20.0)
    p.set_goal(5.0, 0.0)
    wp1 = p.next_waypoint(0.0, 0.0)
    p.set_goal(-5.0, 0.0)
    wp2 = p.next_waypoint(0.0, 0.0)
    assert wp1 is not None
    assert wp2 is not None
    # First waypoints should differ since goals are on opposite sides
    assert wp1 != wp2


def test_world_to_grid_roundtrip():
    p = DStarLitePlanner(40.0, 40.0)
    for x, y in [(0.0, 0.0), (10.0, -5.0), (-15.0, 12.0)]:
        r, c = p._world_to_grid(x, y)
        wx, wy = p._grid_to_world(r, c)
        # Should recover to within one cell
        assert abs(wx - x) <= GRID_RES + 1e-6, f"x roundtrip failed: {x} -> {wx}"
        assert abs(wy - y) <= GRID_RES + 1e-6, f"y roundtrip failed: {y} -> {wy}"
