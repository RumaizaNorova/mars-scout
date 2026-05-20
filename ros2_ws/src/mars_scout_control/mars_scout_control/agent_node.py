"""
mars_scout_control — AgentNode

ROS2 Action Server for NavigateToTarget.
Wires together AgentFSM + PurePursuitController + odometry + VLM targets.

Action:  mars_scout_msgs/action/NavigateToTarget
Topics subscribed:
  /rover/geometry/terrain_target   mars_scout_msgs/TerrainTarget   (3-D waypoints)
  /rover/odom                      nav_msgs/Odometry
Topics published:
  /rover/cmd_vel                   geometry_msgs/Twist
  /rover/state                     mars_scout_msgs/RoverState
"""

from __future__ import annotations
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from mars_scout_msgs.action import NavigateToTarget
from mars_scout_msgs.msg    import TerrainTarget, RoverState

from mars_scout_control.fsm        import AgentFSM, FSMConfig, State
from mars_scout_control.controller import PurePursuitController, ControllerConfig


class AgentNode(Node):

    CONTROL_HZ = 10.0

    def __init__(self):
        super().__init__("agent_node")

        cb_group = ReentrantCallbackGroup()

        # ── State ─────────────────────────────────────────────────────────────
        self._fsm        = AgentFSM()
        self._controller = PurePursuitController()
        self._goal_handle: ServerGoalHandle | None = None

        # Rover pose (from odometry)
        self._rover_x:   float = 0.0
        self._rover_y:   float = 0.0
        self._rover_yaw: float = 0.0

        # Latest 3-D waypoint from geometry node
        self._goal_x:    float = float("nan")
        self._goal_y:    float = float("nan")
        self._latest_target: TerrainTarget | None = None

        # ── Action server ─────────────────────────────────────────────────────
        self._action_server = ActionServer(
            self,
            NavigateToTarget,
            "navigate_to_target",
            execute_callback  = self._execute,
            goal_callback     = self._goal_cb,
            cancel_callback   = self._cancel_cb,
            callback_group    = cb_group,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            TerrainTarget, "/rover/geometry/terrain_target",
            self._cb_target, 10, callback_group=cb_group,
        )
        self.create_subscription(
            Odometry, "/rover/odom",
            self._cb_odom, 10, callback_group=cb_group,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_cmd   = self.create_publisher(Twist,      "/rover/cmd_vel", 10)
        self._pub_state = self.create_publisher(RoverState, "/rover/state",   10)

        self.get_logger().info("AgentNode ready — waiting for NavigateToTarget goals.")

    # ── Action callbacks ──────────────────────────────────────────────────────

    def _goal_cb(self, goal_request) -> GoalResponse:
        if self._fsm.state not in (State.IDLE, State.ARRIVED, State.ABORTED):
            self.get_logger().warn("Rejecting goal — mission already active.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        self.get_logger().info("Cancel requested.")
        self._fsm.cancel()
        return CancelResponse.ACCEPT

    async def _execute(self, goal_handle: ServerGoalHandle):
        goal = goal_handle.request
        self.get_logger().info(
            f"Starting mission: '{goal.query_text}'  "
            f"timeout={goal.timeout_sec}s  "
            f"arrival_r={goal.arrival_radius_m}m"
        )

        self._goal_handle = goal_handle
        self._fsm.reset()
        self._fsm.goal_received(
            timeout_sec      = goal.timeout_sec,
            arrival_radius_m = goal.arrival_radius_m or 0.5,
        )

        dt  = 1.0 / self.CONTROL_HZ
        rate = self.create_rate(self.CONTROL_HZ)

        while rclpy.ok():
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                self._fsm.cancel()

            # Compute distance to current waypoint
            dist = self._distance_to_goal()

            # Get latest VLM data
            found  = False
            conf   = 0.0
            desc   = ""
            if self._latest_target is not None:
                found = self._latest_target.target_found
                conf  = self._latest_target.overall_confidence
                desc  = self._latest_target.vlm_description

            # Tick FSM
            status = self._fsm.update(
                dt=dt,
                target_found=found,
                confidence=conf,
                distance_m=dist,
                description=desc,
            )

            # Compute and publish velocity command
            cmd = self._compute_cmd(status.state)
            self._pub_cmd.publish(cmd)
            self._publish_rover_state(status, goal.query_text)

            # Stream feedback
            feedback = NavigateToTarget.Feedback()
            feedback.fsm_state          = status.state.value
            feedback.distance_to_goal_m = dist if not math.isnan(dist) else -1.0
            feedback.vlm_confidence     = conf
            feedback.vlm_description    = desc
            feedback.rover_position.x   = self._rover_x
            feedback.rover_position.y   = self._rover_y
            goal_handle.publish_feedback(feedback)

            # Terminal states
            if self._fsm.is_terminal:
                break

            rate.sleep()

        # Stop rover
        self._pub_cmd.publish(Twist())

        result = NavigateToTarget.Result()
        result.success           = (self._fsm.state == State.ARRIVED)
        result.outcome           = status.outcome.value if status.outcome else "UNKNOWN"
        result.elapsed_sec       = status.elapsed_sec
        result.final_position.x  = self._rover_x
        result.final_position.y  = self._rover_y
        result.vlm_description   = desc

        if result.success:
            goal_handle.succeed()
            self.get_logger().info(f"ARRIVED in {status.elapsed_sec:.1f}s")
        else:
            goal_handle.abort()
            self.get_logger().warn(f"ABORTED: {result.outcome}")

        self._fsm.reset()
        return result

    # ── Topic callbacks ───────────────────────────────────────────────────────

    def _cb_target(self, msg: TerrainTarget):
        self._latest_target = msg
        # Update goal from 3-D waypoint if available
        if msg.target_found and msg.waypoint.header.frame_id:
            self._goal_x = msg.waypoint.point.x
            self._goal_y = msg.waypoint.point.y

    def _cb_odom(self, msg: Odometry):
        self._rover_x = msg.pose.pose.position.x
        self._rover_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Yaw from quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self._rover_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── Velocity computation ──────────────────────────────────────────────────

    def _compute_cmd(self, state: State) -> Twist:
        cmd = Twist()

        if state == State.SEARCHING:
            cmd.angular.z = 0.35   # slow scan rotation

        elif state in (State.APPROACHING, State.VERIFYING):
            if not math.isnan(self._goal_x):
                out = self._controller.compute(
                    self._rover_x, self._rover_y, self._rover_yaw,
                    self._goal_x,  self._goal_y,
                )
                cmd.linear.x  = out.linear
                cmd.angular.z = out.angular

        # ARRIVED / ABORTED / IDLE → zero velocity (default Twist)
        return cmd

    def _distance_to_goal(self) -> float:
        if math.isnan(self._goal_x):
            return float("nan")
        return math.hypot(self._goal_x - self._rover_x, self._goal_y - self._rover_y)

    # ── State publisher ───────────────────────────────────────────────────────

    def _publish_rover_state(self, status, query_text: str):
        msg = RoverState()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.fsm_state       = status.state.value
        msg.active_query_text  = query_text
        msg.distance_to_goal   = status.distance_to_goal_m if not math.isnan(
            status.distance_to_goal_m) else -1.0
        msg.query_elapsed_sec  = status.elapsed_sec
        self._pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
