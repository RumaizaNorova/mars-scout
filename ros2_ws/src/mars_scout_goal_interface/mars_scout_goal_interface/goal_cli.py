"""
Mars Scout — Goal Interface CLI

Usage
-----
One-shot:
    ros2 run mars_scout_goal_interface goal_cli "find the rock that looks like a skull"

Interactive REPL:
    ros2 run mars_scout_goal_interface goal_cli

Commands in interactive mode:
    > find the skull-shaped rock        send a navigation goal
    > cancel                            cancel active goal
    > status                            print current rover state
    > quit / exit                       exit
"""

from __future__ import annotations
import sys
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle

from mars_scout_msgs.action import NavigateToTarget
from mars_scout_msgs.msg    import TerrainQuery, RoverState

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_GREY   = "\033[90m"

def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_RESET}"


# ── State colour map ──────────────────────────────────────────────────────────
_STATE_COLOUR = {
    "IDLE":        _GREY,
    "SEARCHING":   _YELLOW,
    "APPROACHING": _CYAN,
    "VERIFYING":   _CYAN,
    "ARRIVED":     _GREEN,
    "ABORTED":     _RED,
}


class GoalCLI(Node):

    def __init__(self):
        super().__init__("goal_cli")

        self._action_client = ActionClient(self, NavigateToTarget, "navigate_to_target")
        self._query_pub     = self.create_publisher(TerrainQuery, "/rover/vlm/query", 10)
        self._goal_handle:  ClientGoalHandle | None = None
        self._latest_state: RoverState | None = None

        self.create_subscription(RoverState, "/rover/state", self._cb_state, 10)

    # ── Send goal ─────────────────────────────────────────────────────────────

    def send_goal(
        self,
        query_text:      str,
        timeout_sec:     float = 120.0,
        arrival_radius_m: float = 0.5,
        min_confidence:  float = 0.4,
    ):
        print(_c(_BOLD, f"\n🚀  Sending goal: ") + _c(_CYAN, f'"{query_text}"'))

        # Also publish the query so the perception node updates immediately
        q_msg = TerrainQuery()
        q_msg.header.stamp = self.get_clock().now().to_msg()
        q_msg.query_text   = query_text
        q_msg.timeout_sec  = timeout_sec
        self._query_pub.publish(q_msg)

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            print(_c(_RED, "✗  Action server not available. Is agent_node running?"))
            return

        goal = NavigateToTarget.Goal()
        goal.query_text       = query_text
        goal.timeout_sec      = timeout_sec
        goal.arrival_radius_m = arrival_radius_m
        goal.min_confidence   = min_confidence

        send_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_cb,
        )
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle: ClientGoalHandle = future.result()
        if not handle.accepted:
            print(_c(_RED, "✗  Goal rejected by server."))
            return
        self._goal_handle = handle
        print(_c(_GREEN, "✓  Goal accepted — rover is on its way.\n"))
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb   = feedback_msg.feedback
        state = fb.fsm_state
        colour = _STATE_COLOUR.get(state, _RESET)
        dist  = f"{fb.distance_to_goal_m:.2f}m" if fb.distance_to_goal_m >= 0 else "—"
        conf  = f"{fb.vlm_confidence:.0%}"
        desc  = fb.vlm_description[:55] + "…" if len(fb.vlm_description) > 55 else fb.vlm_description
        print(
            f"  {_c(colour, f'[{state:<11}]')}  "
            f"dist={_c(_BOLD, dist):>8}  "
            f"conf={conf}  "
            f"{_c(_GREY, desc)}"
        )

    def _result_cb(self, future):
        result = future.result().result
        if result.success:
            print(_c(_GREEN, f"\n✓  ARRIVED  ({result.elapsed_sec:.1f}s)"))
            print(_c(_GREY, f"   VLM: {result.vlm_description}"))
        else:
            print(_c(_RED, f"\n✗  {result.outcome}  ({result.elapsed_sec:.1f}s)"))
        print()
        self._goal_handle = None

    def _cb_state(self, msg: RoverState):
        self._latest_state = msg

    # ── Cancel ────────────────────────────────────────────────────────────────

    def cancel(self):
        if self._goal_handle is None:
            print(_c(_YELLOW, "No active goal to cancel."))
            return
        print(_c(_YELLOW, "Cancelling…"))
        self._goal_handle.cancel_goal_async()

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self):
        if self._latest_state is None:
            print(_c(_GREY, "No rover state received yet."))
            return
        s = self._latest_state
        colour = _STATE_COLOUR.get(s.fsm_state, _RESET)
        print(
            f"  State: {_c(colour, s.fsm_state)}  "
            f"query: '{s.active_query_text}'  "
            f"dist: {s.distance_to_goal:.2f}m  "
            f"elapsed: {s.query_elapsed_sec:.1f}s"
        )


# ── REPL ──────────────────────────────────────────────────────────────────────

def _print_banner():
    print(_c(_BOLD + _CYAN, """
╔══════════════════════════════════════════╗
║        🔴  MARS SCOUT  🔴                ║
║   Vision-Language Rover Navigation CLI   ║
╚══════════════════════════════════════════╝
"""))
    print("  Commands:")
    print("    <query text>   — send a navigation goal")
    print("    cancel         — abort current mission")
    print("    status         — show rover state")
    print("    quit           — exit")
    print()


def _spin_thread(node: Node):
    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)
    node = GoalCLI()

    spin_thread = threading.Thread(target=_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    one_shot = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    if one_shot:
        node.send_goal(one_shot)
        # Wait for result (spin handles callbacks in background thread)
        try:
            spin_thread.join(timeout=130.0)
        except KeyboardInterrupt:
            node.cancel()
    else:
        _print_banner()
        try:
            while True:
                try:
                    user_input = input(_c(_BOLD, "mars-scout> ")).strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                cmd = user_input.lower()

                if cmd in ("quit", "exit", "q"):
                    node.cancel()
                    break
                elif cmd == "cancel":
                    node.cancel()
                elif cmd == "status":
                    node.status()
                elif cmd.startswith("help"):
                    _print_banner()
                else:
                    node.send_goal(user_input)

        except KeyboardInterrupt:
            node.cancel()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
