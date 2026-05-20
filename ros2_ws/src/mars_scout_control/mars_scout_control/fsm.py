"""
Rover agent finite-state machine — pure Python, zero ROS dependencies.

States
------
IDLE          Waiting for a NavigateToTarget goal.
SEARCHING     Rotating to scan for the target; no confirmed detection yet.
APPROACHING   Target acquired; driving toward 3-D waypoint.
VERIFYING     Rover is close — double-checking target before declaring arrival.
ARRIVED       Target reached. Terminal success state.
ABORTED       Goal cancelled or timed out. Terminal failure state.

Transitions
-----------
IDLE        → SEARCHING    on goal_received()
SEARCHING   → APPROACHING  on target_acquired(confidence ≥ threshold)
SEARCHING   → ABORTED      on timeout() or cancel()
APPROACHING → VERIFYING    on within_arrival_radius()
APPROACHING → SEARCHING    on target_lost()  (N consecutive frames with no detection)
APPROACHING → ABORTED      on timeout() or cancel()
VERIFYING   → ARRIVED      on verified()     (M consecutive frames still close)
VERIFYING   → APPROACHING  on target_lost()  (false positive — back off)
VERIFYING   → ABORTED      on timeout() or cancel()

All logic here is deterministic and has no side-effects,
making it straightforward to unit-test without any mocking.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class State(str, Enum):
    IDLE        = "IDLE"
    SEARCHING   = "SEARCHING"
    APPROACHING = "APPROACHING"
    VERIFYING   = "VERIFYING"
    ARRIVED     = "ARRIVED"
    ABORTED     = "ABORTED"


class Outcome(str, Enum):
    ARRIVED         = "ARRIVED"
    TIMEOUT         = "TIMEOUT"
    CANCELLED       = "CANCELLED"
    LOW_CONFIDENCE  = "LOW_CONFIDENCE"


@dataclass
class FSMConfig:
    min_confidence:         float = 0.45   # to transition SEARCHING → APPROACHING
    arrival_radius_m:       float = 0.5    # metres — triggers VERIFYING
    verify_frames:          int   = 5      # consecutive close frames to declare ARRIVED
    lost_frames_threshold:  int   = 8      # consecutive no-detection frames → SEARCHING
    timeout_sec:            float = 120.0  # 0 = unlimited


@dataclass
class FSMStatus:
    """Snapshot of FSM state — sent as action feedback."""
    state:              State
    distance_to_goal_m: float   = float("nan")
    vlm_confidence:     float   = 0.0
    vlm_description:    str     = ""
    elapsed_sec:        float   = 0.0
    outcome:            Optional[Outcome] = None


class AgentFSM:
    """
    Deterministic FSM for rover target navigation.
    Call update() on every control tick; it returns the new FSMStatus.
    """

    def __init__(self, cfg: FSMConfig | None = None):
        self.cfg    = cfg or FSMConfig()
        self.state  = State.IDLE
        self._elapsed:       float = 0.0
        self._lost_count:    int   = 0
        self._verify_count:  int   = 0
        self._outcome:       Optional[Outcome] = None
        self._description:   str   = ""
        self._confidence:    float = 0.0
        self._distance_m:    float = float("nan")

    # ── External events ───────────────────────────────────────────────────────

    def goal_received(self, timeout_sec: float = 0.0, arrival_radius_m: float = 0.0):
        """Operator sent a NavigateToTarget goal."""
        assert self.state == State.IDLE, f"Cannot accept goal in state {self.state}"
        if timeout_sec > 0:
            self.cfg.timeout_sec = timeout_sec
        if arrival_radius_m > 0:
            self.cfg.arrival_radius_m = arrival_radius_m
        self._elapsed      = 0.0
        self._lost_count   = 0
        self._verify_count = 0
        self._outcome      = None
        self._transition(State.SEARCHING)

    def cancel(self):
        """Operator cancelled the goal."""
        if self.state not in (State.ARRIVED, State.ABORTED):
            self._outcome = Outcome.CANCELLED
            self._transition(State.ABORTED)

    def reset(self):
        """Return to IDLE so a new goal can be accepted."""
        self.state    = State.IDLE
        self._outcome = None

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        dt:           float,
        target_found: bool,
        confidence:   float,
        distance_m:   float,
        description:  str = "",
    ) -> FSMStatus:
        """
        Advance the FSM by one control tick.

        Parameters
        ----------
        dt            : seconds since last tick
        target_found  : did the VLM find the target this frame?
        confidence    : VLM confidence [0, 1]
        distance_m    : distance to current waypoint in metres (NaN if unknown)
        description   : latest VLM description string

        Returns
        -------
        FSMStatus
        """
        self._elapsed    += dt
        self._distance_m  = distance_m
        self._confidence  = confidence
        self._description = description

        # Terminal states — nothing to do
        if self.state in (State.ARRIVED, State.ABORTED):
            return self._status()

        # Global timeout check
        if self.cfg.timeout_sec > 0 and self._elapsed >= self.cfg.timeout_sec:
            self._outcome = Outcome.TIMEOUT
            self._transition(State.ABORTED)
            return self._status()

        # State-specific logic
        if self.state == State.SEARCHING:
            self._tick_searching(target_found, confidence)

        elif self.state == State.APPROACHING:
            self._tick_approaching(target_found, confidence, distance_m)

        elif self.state == State.VERIFYING:
            self._tick_verifying(target_found, distance_m)

        return self._status()

    # ── State tick methods ────────────────────────────────────────────────────

    def _tick_searching(self, found: bool, conf: float):
        if found and conf >= self.cfg.min_confidence:
            self._lost_count = 0
            self._transition(State.APPROACHING)

    def _tick_approaching(self, found: bool, conf: float, dist: float):
        if not found:
            self._lost_count += 1
            if self._lost_count >= self.cfg.lost_frames_threshold:
                self._lost_count = 0
                self._transition(State.SEARCHING)
            return

        self._lost_count = 0

        import math
        if not math.isnan(dist) and dist <= self.cfg.arrival_radius_m:
            self._verify_count = 0
            self._transition(State.VERIFYING)

    def _tick_verifying(self, found: bool, dist: float):
        import math
        if not found or (not math.isnan(dist) and dist > self.cfg.arrival_radius_m * 1.5):
            # False positive — back to approaching
            self._verify_count = 0
            self._transition(State.APPROACHING)
            return

        self._verify_count += 1
        if self._verify_count >= self.cfg.verify_frames:
            self._outcome = Outcome.ARRIVED
            self._transition(State.ARRIVED)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new_state: State):
        self.state = new_state

    def _status(self) -> FSMStatus:
        return FSMStatus(
            state              = self.state,
            distance_to_goal_m = self._distance_m,
            vlm_confidence     = self._confidence,
            vlm_description    = self._description,
            elapsed_sec        = self._elapsed,
            outcome            = self._outcome,
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in (State.ARRIVED, State.ABORTED)
