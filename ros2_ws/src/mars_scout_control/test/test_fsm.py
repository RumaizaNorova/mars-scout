import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_control.fsm import AgentFSM, FSMConfig, State, Outcome


def make_fsm(**kwargs) -> AgentFSM:
    cfg = FSMConfig(**kwargs)
    fsm = AgentFSM(cfg)
    return fsm


# ── Basic lifecycle ───────────────────────────────────────────────────────────

def test_initial_state_is_idle():
    assert make_fsm().state == State.IDLE


def test_goal_received_transitions_to_searching():
    fsm = make_fsm()
    fsm.goal_received()
    assert fsm.state == State.SEARCHING


def test_cannot_accept_goal_when_active():
    fsm = make_fsm()
    fsm.goal_received()
    with pytest.raises(AssertionError):
        fsm.goal_received()


def test_cancel_from_searching():
    fsm = make_fsm()
    fsm.goal_received()
    fsm.cancel()
    assert fsm.state == State.ABORTED
    assert fsm.is_terminal


# ── SEARCHING → APPROACHING ───────────────────────────────────────────────────

def test_high_confidence_target_triggers_approaching():
    fsm = make_fsm(min_confidence=0.4)
    fsm.goal_received()
    status = fsm.update(dt=0.1, target_found=True, confidence=0.8,
                        distance_m=5.0, description="rock ahead")
    assert status.state == State.APPROACHING


def test_low_confidence_stays_searching():
    fsm = make_fsm(min_confidence=0.6)
    fsm.goal_received()
    status = fsm.update(dt=0.1, target_found=True, confidence=0.3,
                        distance_m=5.0)
    assert status.state == State.SEARCHING


def test_no_target_stays_searching():
    fsm = make_fsm()
    fsm.goal_received()
    for _ in range(10):
        status = fsm.update(dt=0.1, target_found=False, confidence=0.0, distance_m=5.0)
    assert status.state == State.SEARCHING


# ── APPROACHING → VERIFYING → ARRIVED ────────────────────────────────────────

def test_full_happy_path():
    """IDLE → SEARCHING → APPROACHING → VERIFYING → ARRIVED."""
    fsm = make_fsm(min_confidence=0.4, arrival_radius_m=1.0, verify_frames=3)
    fsm.goal_received()

    # Find target
    fsm.update(dt=0.1, target_found=True, confidence=0.8, distance_m=5.0)
    assert fsm.state == State.APPROACHING

    # Approach until within radius
    fsm.update(dt=0.1, target_found=True, confidence=0.8, distance_m=0.8)
    assert fsm.state == State.VERIFYING

    # Verify for required frames
    for _ in range(3):
        status = fsm.update(dt=0.1, target_found=True, confidence=0.8, distance_m=0.8)

    assert status.state == State.ARRIVED
    assert status.outcome == Outcome.ARRIVED
    assert fsm.is_terminal


# ── Target lost → back to SEARCHING ──────────────────────────────────────────

def test_target_lost_returns_to_searching():
    fsm = make_fsm(min_confidence=0.4, lost_frames_threshold=3)
    fsm.goal_received()
    fsm.update(dt=0.1, target_found=True, confidence=0.8, distance_m=5.0)
    assert fsm.state == State.APPROACHING

    for _ in range(3):
        status = fsm.update(dt=0.1, target_found=False, confidence=0.0, distance_m=5.0)
    assert status.state == State.SEARCHING


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout_aborts():
    fsm = make_fsm(timeout_sec=1.0)
    fsm.goal_received(timeout_sec=1.0)
    # Tick past timeout
    status = None
    for _ in range(15):
        status = fsm.update(dt=0.1, target_found=False, confidence=0.0, distance_m=5.0)
    assert status.state == State.ABORTED
    assert status.outcome == Outcome.TIMEOUT


def test_elapsed_time_accumulates():
    fsm = make_fsm()
    fsm.goal_received()
    for _ in range(10):
        status = fsm.update(dt=0.1, target_found=False, confidence=0.0, distance_m=5.0)
    assert status.elapsed_sec == pytest.approx(1.0, abs=1e-6)


# ── Reset and reuse ───────────────────────────────────────────────────────────

def test_reset_allows_new_goal():
    fsm = make_fsm()
    fsm.goal_received()
    fsm.cancel()
    fsm.reset()
    assert fsm.state == State.IDLE
    fsm.goal_received()   # should not raise
    assert fsm.state == State.SEARCHING


# ── Terminal state is inert ───────────────────────────────────────────────────

def test_arrived_state_is_inert():
    fsm = make_fsm(min_confidence=0.4, arrival_radius_m=1.0, verify_frames=1)
    fsm.goal_received()
    fsm.update(dt=0.1, target_found=True, confidence=0.9, distance_m=5.0)
    fsm.update(dt=0.1, target_found=True, confidence=0.9, distance_m=0.5)
    fsm.update(dt=0.1, target_found=True, confidence=0.9, distance_m=0.5)
    assert fsm.state == State.ARRIVED
    # More ticks should not change state
    for _ in range(10):
        status = fsm.update(dt=0.1, target_found=True, confidence=0.9, distance_m=0.5)
    assert status.state == State.ARRIVED
