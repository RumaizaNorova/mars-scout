"""
D* Lite global path planner on a traversability grid.

D* Lite (Koenig & Likhachev 2002) is an incremental heuristic search
algorithm that plans shortest paths from the rover's current position to a
goal on a cost grid. Unlike A* it can replan efficiently when the grid
changes (new obstacle detected, VLM confidence update) without restarting
from scratch -- only the affected nodes are re-expanded.

Why D* Lite over Pure Pursuit alone
------------------------------------
Pure Pursuit is a reactive controller: it drives straight toward the
detected target waypoint with no global awareness. On Mars terrain this
causes the rover to drive into craters, over loose lag pavement, or up
steep slopes that flip or strand it. D* Lite adds one forward-planning
step: before committing to a waypoint, plan a collision-free path and
feed the first waypoint on that path to Pure Pursuit.

Grid model
----------
The traversability grid is a 2D float32 array (cost per cell). Each cell
represents a GRID_RES x GRID_RES metre square of terrain.

Cost values:
  0.0         free traversable ground
  0.0 - 1.0   traversable with penalty (slope, soft regolith, lag pavement)
  inf         obstacle (crater interior, steep slope > MAX_SLOPE_DEG)

Terrain cost heuristics (from the hirise_terrain_builder elevation data):
  slope > 20 deg     ->  inf         (Golombek 2018: tipover threshold)
  slope 10-20 deg    ->  2.0 * base  (high cost, avoid)
  slope  5-10 deg    ->  1.3 * base  (moderate cost)
  slope < 5  deg     ->  1.0 * base  (nominal)
  crater interior    ->  3.0 * base  (prefer to avoid but not impassable)

Inflation
---------
Obstacle cells are inflated by INFLATION_RADIUS to give the rover
clearance (rover half-width ~ 1.4 m, so inflate 2 cells at 0.5 m/cell).

Interface
---------
  planner = DStarLitePlanner(width_m, depth_m)
  planner.update_from_elevation(elevation_array, terrain_w, terrain_d)
  planner.set_goal(goal_x, goal_y)
  waypoint = planner.next_waypoint(rover_x, rover_y)

Call next_waypoint() every FSM tick. It returns the next (x, y) position
along the planned path to feed to PurePursuitController.compute().

If no path exists (rover is trapped), returns None.

References
----------
Koenig, S. & Likhachev, M. (2002). D* Lite. AAAI-02.
Golombek et al. 2018, JGR Planets: Mars terrain traversability model.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Grid parameters ────────────────────────────────────────────────────────────
GRID_RES        = 0.5    # metres per cell
MAX_SLOPE_DEG   = 20.0   # above this: impassable (Golombek 2018 tipover limit)
INFLATION_RADIUS = 2     # cells (1.0 m at GRID_RES=0.5)
INF             = float("inf")

# 8-connected neighbours: (dr, dc, move_cost_multiplier)
# Diagonals cost sqrt(2) more than cardinal moves
_NEIGHBOURS = [
    (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
    (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
]


@dataclass
class _Node:
    """Priority queue entry -- separate from the grid to avoid stale entries."""
    k1: float
    k2: float
    row: int
    col: int

    def __lt__(self, other: "_Node") -> bool:
        return (self.k1, self.k2) < (other.k1, other.k2)


class DStarLitePlanner:
    """
    D* Lite planner on a 2D traversability cost grid.

    Coordinate conventions
    ----------------------
    World (x, y) in metres, origin at scene centre.
    Grid (row, col): row 0 = most-negative Y, col 0 = most-negative X.
    """

    def __init__(self, width_m: float, depth_m: float) -> None:
        self._width_m = width_m
        self._depth_m = depth_m
        self._nx = max(4, int(math.ceil(width_m / GRID_RES)))
        self._ny = max(4, int(math.ceil(depth_m / GRID_RES)))

        # Cost grid: free by default
        self._cost: np.ndarray = np.ones((self._ny, self._nx), dtype=np.float32)

        # D* Lite state
        self._g:   Dict[Tuple[int,int], float] = {}  # g-values
        self._rhs: Dict[Tuple[int,int], float] = {}  # rhs-values
        self._open: List[_Node] = []                 # priority queue
        self._in_open: Dict[Tuple[int,int], float] = {}  # key -> k1 for stale check

        self._goal: Optional[Tuple[int,int]] = None
        self._start: Optional[Tuple[int,int]] = None
        self._km: float = 0.0  # heuristic correction for moved start

        self._path: List[Tuple[float,float]] = []  # cached world-coord path
        self._path_dirty: bool = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_from_elevation(
        self,
        elevation: np.ndarray,
        terrain_w: float,
        terrain_d: float,
    ) -> None:
        """
        Rebuild the cost grid from a (ny, nx) elevation array (metres).

        Computes slope from central finite differences, maps slope to cost,
        then inflates obstacle cells by INFLATION_RADIUS.

        Call this once at scene load. Call update_cell() for incremental
        changes (e.g., newly detected crater, soft-soil zone).
        """
        elev_res_x = terrain_w / max(elevation.shape[1] - 1, 1)
        elev_res_y = terrain_d / max(elevation.shape[0] - 1, 1)

        # Gradient via central differences
        dy, dx = np.gradient(elevation.astype(np.float64), elev_res_y, elev_res_x)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        # Resample slope to grid resolution via nearest-neighbour
        from scipy.ndimage import zoom as _zoom
        scale_y = self._ny / slope_deg.shape[0]
        scale_x = self._nx / slope_deg.shape[1]
        slope_grid = _zoom(slope_deg, (scale_y, scale_x), order=0).astype(np.float32)
        slope_grid = slope_grid[:self._ny, :self._nx]

        # Slope -> cost
        cost = np.where(slope_grid > MAX_SLOPE_DEG, INF,
               np.where(slope_grid > 10.0, 2.0,
               np.where(slope_grid >  5.0, 1.3,
                                           1.0))).astype(np.float32)

        # Inflate obstacles
        cost = _inflate_obstacles(cost, INFLATION_RADIUS)

        self._cost = cost
        self._path_dirty = True
        self._reset_dstar()

    def update_cell(self, world_x: float, world_y: float, new_cost: float) -> None:
        """
        Update a single cell cost (e.g., newly detected obstacle).
        Triggers incremental D* Lite replanning.
        """
        r, c = self._world_to_grid(world_x, world_y)
        if not self._in_bounds(r, c):
            return
        old_cost = float(self._cost[r, c])
        if abs(old_cost - new_cost) < 1e-6:
            return
        self._cost[r, c] = new_cost

        # Update all nodes whose rhs depends on this cell
        for dr, dc, _ in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                self._update_vertex((nr, nc))
        self._update_vertex((r, c))
        self._path_dirty = True

    def set_goal(self, goal_x: float, goal_y: float) -> None:
        """Set or update the goal position in world coordinates."""
        gr, gc = self._world_to_grid(goal_x, goal_y)
        gr = max(0, min(self._ny - 1, gr))
        gc = max(0, min(self._nx - 1, gc))
        new_goal = (gr, gc)

        if new_goal == self._goal:
            return

        self._goal = new_goal
        self._path_dirty = True
        self._reset_dstar()

    def next_waypoint(
        self,
        rover_x: float,
        rover_y: float,
        lookahead_m: float = 2.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Return the next (x, y) waypoint along the planned path.

        The waypoint is the first path node at least `lookahead_m` ahead
        of the rover, or the goal itself if the path is short.

        Returns None if no path exists (rover trapped by obstacles).
        """
        if self._goal is None:
            return None

        sr, sc = self._world_to_grid(rover_x, rover_y)
        sr = max(0, min(self._ny - 1, sr))
        sc = max(0, min(self._nx - 1, sc))
        new_start = (sr, sc)

        if new_start != self._start:
            # Update km correction when start moves (D* Lite paper, line 26)
            if self._start is not None:
                self._km += self._h(self._start, new_start)
            self._start = new_start
            self._path_dirty = True

        if self._path_dirty:
            self._compute_shortest_path()
            self._path = self._extract_path()
            self._path_dirty = False

        if not self._path:
            return None

        # Find first waypoint beyond lookahead distance
        lookahead_cells = lookahead_m / GRID_RES
        rx, ry = self._grid_to_world(sr, sc)
        for (wx, wy) in self._path:
            if math.hypot(wx - rx, wy - ry) >= lookahead_m:
                return (wx, wy)

        # All path nodes within lookahead -- return goal directly
        return self._grid_to_world(*self._goal)

    def path_cost(self) -> float:
        """Return the total cost of the current path (sum of cell costs along path)."""
        if self._start is None or self._goal is None:
            return INF
        return self._g.get(self._start, INF)

    def has_path(self) -> bool:
        return self.path_cost() < INF

    # ── D* Lite internals ──────────────────────────────────────────────────────

    def _reset_dstar(self) -> None:
        self._g.clear()
        self._rhs.clear()
        self._open.clear()
        self._in_open.clear()
        self._km = 0.0
        self._path = []
        self._path_dirty = True

        if self._goal is not None:
            self._rhs[self._goal] = 0.0
            self._push(self._goal, self._key(self._goal))

    def _key(self, node: Tuple[int,int]) -> Tuple[float, float]:
        g   = self._g.get(node, INF)
        rhs = self._rhs.get(node, INF)
        h   = self._h(self._start, node) if self._start else 0.0
        return (min(g, rhs) + h + self._km, min(g, rhs))

    def _h(self, a: Optional[Tuple[int,int]], b: Tuple[int,int]) -> float:
        """Octile distance heuristic (consistent, admissible for 8-connected grid)."""
        if a is None:
            return 0.0
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return (dr + dc) + (1.414 - 2.0) * min(dr, dc)

    def _push(self, node: Tuple[int,int], key: Tuple[float,float]) -> None:
        entry = _Node(key[0], key[1], node[0], node[1])
        heapq.heappush(self._open, entry)
        self._in_open[node] = key[0]

    def _top_key(self) -> Tuple[float,float]:
        while self._open:
            top = self._open[0]
            node = (top.row, top.col)
            # Stale entry: skip
            if self._in_open.get(node, -1) != top.k1:
                heapq.heappop(self._open)
                continue
            return (top.k1, top.k2)
        return (INF, INF)

    def _pop(self) -> Optional[Tuple[int,int]]:
        while self._open:
            top = heapq.heappop(self._open)
            node = (top.row, top.col)
            if self._in_open.get(node, -1) != top.k1:
                continue
            del self._in_open[node]
            return node
        return None

    def _update_vertex(self, node: Tuple[int,int]) -> None:
        if node != self._goal:
            # rhs = min cost one-step from any successor
            min_rhs = INF
            for dr, dc, move_mul in _NEIGHBOURS:
                nr, nc = node[0] + dr, node[1] + dc
                if self._in_bounds(nr, nc):
                    succ_cost = float(self._cost[nr, nc])
                    if succ_cost < INF:
                        g_succ = self._g.get((nr, nc), INF)
                        candidate = move_mul * GRID_RES * succ_cost + g_succ
                        if candidate < min_rhs:
                            min_rhs = candidate
            self._rhs[node] = min_rhs

        g   = self._g.get(node, INF)
        rhs = self._rhs.get(node, INF)
        # Remove from open (mark stale)
        if node in self._in_open:
            del self._in_open[node]

        if g != rhs:
            self._push(node, self._key(node))

    def _compute_shortest_path(self) -> None:
        if self._start is None or self._goal is None:
            return

        MAX_ITER = self._nx * self._ny * 2
        for _ in range(MAX_ITER):
            k_old = self._top_key()
            k_start = self._key(self._start)
            g_s   = self._g.get(self._start, INF)
            rhs_s = self._rhs.get(self._start, INF)

            if k_old >= k_start and g_s == rhs_s:
                break  # path is consistent

            node = self._pop()
            if node is None:
                break

            g   = self._g.get(node, INF)
            rhs = self._rhs.get(node, INF)

            if g > rhs:
                # Overconsistent: improve g
                self._g[node] = rhs
                for dr, dc, _ in _NEIGHBOURS:
                    pr, pc = node[0] + dr, node[1] + dc
                    if self._in_bounds(pr, pc):
                        self._update_vertex((pr, pc))
            else:
                # Underconsistent: remove g improvement
                self._g[node] = INF
                self._update_vertex(node)
                for dr, dc, _ in _NEIGHBOURS:
                    pr, pc = node[0] + dr, node[1] + dc
                    if self._in_bounds(pr, pc):
                        self._update_vertex((pr, pc))

    def _extract_path(self) -> List[Tuple[float,float]]:
        """
        Greedy path extraction: from start, always step to the lowest-cost
        neighbour until goal is reached or no improvement is possible.
        """
        if self._start is None or self._goal is None:
            return []
        if self._g.get(self._start, INF) == INF:
            return []  # no path

        path = []
        current = self._start
        visited = {current}
        MAX_STEPS = self._nx * self._ny

        for _ in range(MAX_STEPS):
            if current == self._goal:
                path.append(self._grid_to_world(*current))
                break

            best_cost = INF
            best_node = None
            for dr, dc, move_mul in _NEIGHBOURS:
                nr, nc = current[0] + dr, current[1] + dc
                if not self._in_bounds(nr, nc):
                    continue
                nb = (nr, nc)
                if nb in visited:
                    continue
                cell_cost = float(self._cost[nr, nc])
                if cell_cost == INF:
                    continue
                g_nb = self._g.get(nb, INF)
                total = move_mul * GRID_RES * cell_cost + g_nb
                if total < best_cost:
                    best_cost = total
                    best_node = nb

            if best_node is None:
                break  # trapped

            path.append(self._grid_to_world(*current))
            visited.add(best_node)
            current = best_node

        return path

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def _world_to_grid(self, x: float, y: float) -> Tuple[int,int]:
        col = int((x + self._width_m / 2.0) / GRID_RES)
        row = int((y + self._depth_m / 2.0) / GRID_RES)
        return row, col

    def _grid_to_world(self, row: int, col: int) -> Tuple[float,float]:
        x = col * GRID_RES - self._width_m / 2.0 + GRID_RES / 2.0
        y = row * GRID_RES - self._depth_m / 2.0 + GRID_RES / 2.0
        return x, y

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._ny and 0 <= c < self._nx


# ── Obstacle inflation ─────────────────────────────────────────────────────────

def _inflate_obstacles(cost: np.ndarray, radius: int) -> np.ndarray:
    """
    Any cell within `radius` cells of an INF obstacle becomes INF.
    Uses a simple morphological dilation via sliding window.
    """
    if radius <= 0:
        return cost
    result = cost.copy()
    obstacle = np.isinf(cost)

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) > radius + 0.5:
                continue
            shifted = np.roll(np.roll(obstacle, dr, axis=0), dc, axis=1)
            # Zero out wrap-around edges
            if dr > 0:
                shifted[:dr, :] = False
            elif dr < 0:
                shifted[dr:, :] = False
            if dc > 0:
                shifted[:, :dc] = False
            elif dc < 0:
                shifted[:, dc:] = False
            result[shifted] = INF

    return result
