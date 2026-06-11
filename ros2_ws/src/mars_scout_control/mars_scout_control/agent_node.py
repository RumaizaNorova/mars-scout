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

from mars_scout_control.fsm          import AgentFSM, FSMConfig, State
from mars_scout_control.controller   import PurePursuitController, ControllerConfig
from mars_scout_control.dstar_planner import DStarLitePlanner

# ── Ground control telemetry (optional — disabled if env vars not set) ─────────
try:
    from ground_control import TerrainMemory, VLMQualityMonitor, FleetMonitor, MissionPlanner
    _GC_AVAILABLE = True
except Exception as _gc_err:
    print(f"[agent_node] Ground control telemetry unavailable: {_gc_err}\n"
          f"  Set MONGODB_URI, ELASTICSEARCH_URL, PHOENIX_COLLECTOR_ENDPOINT "
          f"to enable it.")
    _GC_AVAILABLE = False


class AgentNode(Node):

    CONTROL_HZ    = 10.0
    LOG_EVERY_N   = 5   # log observation every N ticks (~2 Hz at 10 Hz control)

    def __init__(self):
        super().__init__("agent_node")

        cb_group = ReentrantCallbackGroup()

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("terrain_width_m", 40.0)
        self.declare_parameter("terrain_depth_m", 40.0)
        self.declare_parameter("use_dstar", True)
        terrain_w = self.get_parameter("terrain_width_m").value
        terrain_d = self.get_parameter("terrain_depth_m").value
        self._use_dstar: bool = self.get_parameter("use_dstar").value

        # ── State ─────────────────────────────────────────────────────────────
        self._fsm        = AgentFSM()
        self._controller = PurePursuitController()
        self._dstar      = DStarLitePlanner(terrain_w, terrain_d)
        self._goal_handle: ServerGoalHandle | None = None
        self._tick:        int = 0
        self._mission_id:  str | None = None
        self._prev_state:  str = State.IDLE.value

        # Rover pose (from odometry)
        self._rover_x:   float = 0.0
        self._rover_y:   float = 0.0
        self._rover_yaw: float = 0.0

        # Latest 3-D waypoint from geometry node (VLM goal)
        self._goal_x:    float = float("nan")
        self._goal_y:    float = float("nan")
        # D* Lite planned waypoint (intermediate target for Pure Pursuit)
        self._planned_wp_x: float = float("nan")
        self._planned_wp_y: float = float("nan")
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

        # ── Ground control telemetry ──────────────────────────────────────────
        self._planner: MissionPlanner | None = None
        if _GC_AVAILABLE:
            try:
                mem   = TerrainMemory()
                qual  = VLMQualityMonitor()
                fleet = FleetMonitor()
                self._planner = MissionPlanner(
                    mem, qual, fleet,
                    vlm_backend_name="gemini/gemini-1.5-flash",
                )
                self.get_logger().info("Ground control telemetry: MongoDB + Arize + Elastic active.")
            except Exception as exc:
                self.get_logger().error(
                    f"Ground control init failed: {exc}\n"
                    "  Telemetry disabled for this session."
                )
                self._planner = None

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
        self._tick        = 0

        # ── Plan mission (queries terrain catalog + calibration) ──────────────
        plan = None
        if self._planner is not None:
            try:
                plan = self._planner.plan_mission(
                    query_text   = goal.query_text,
                    start_pos    = (self._rover_x, self._rover_y),
                    terrain_unit = getattr(goal, "terrain_unit", "unknown"),
                )
                self._mission_id = plan.mission_id
                self.get_logger().info(
                    f"[GC] Mission {plan.mission_id[:8]}  "
                    f"rec_conf={plan.recommended_confidence:.2f}  "
                    f"seeded_goal={plan.seeded_goal}  "
                    f"nearby_features={len(plan.known_features_nearby)}"
                )
                # Seed waypoint from historical missions if available
                if plan.seeded_goal and math.isnan(self._goal_x):
                    self._goal_x, self._goal_y = plan.seeded_goal
            except Exception as exc:
                self.get_logger().error(f"[GC] plan_mission failed: {exc}")

        # Apply recommended confidence threshold from calibration history
        rec_conf = plan.recommended_confidence if plan else 0.45
        self._fsm.reset()
        self._fsm.cfg.min_confidence = rec_conf
        self._fsm.goal_received(
            timeout_sec      = (plan.timeout_hint_sec if plan else goal.timeout_sec) or goal.timeout_sec,
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

            # Detect FSM state transitions → ground control (false positive tracking)
            curr_state = status.state.value
            if curr_state != self._prev_state and self._planner and self._mission_id:
                try:
                    self._planner.record_fsm_transition(
                        self._mission_id, self._prev_state, curr_state
                    )
                except Exception:
                    pass
            self._prev_state = curr_state

            # Log observation to all three telemetry systems (every N ticks)
            self._tick += 1
            if self._planner and self._mission_id and self._tick % self.LOG_EVERY_N == 0:
                obs = {
                    "seq":               self._tick,
                    "timestamp":         None,   # planner fills this
                    "fsm_state":         status.state.value,
                    "target_found":      found,
                    "confidence":        conf,
                    "description":       desc,
                    "distance_to_goal_m":dist if not math.isnan(dist) else -1.0,
                    "rover_position":    {"x": self._rover_x, "y": self._rover_y},
                    "inference_ms":      getattr(self._latest_target, "inference_ms", 0.0)
                                         if self._latest_target else 0.0,
                    "query_text":        goal.query_text,
                    "vlm_backend":       "gemini/gemini-1.5-flash",
                }
                try:
                    self._planner.log_tick(self._mission_id, obs)
                except Exception:
                    pass

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

        # Close mission in all three telemetry systems (non-blocking)
        if self._planner and self._mission_id:
            try:
                self._planner.close_mission(
                    mission_id     = self._mission_id,
                    outcome        = result.outcome,
                    elapsed_sec    = status.elapsed_sec,
                    final_position = (self._rover_x, self._rover_y),
                )
            except Exception as exc:
                self.get_logger().error(f"[GC] close_mission failed: {exc}")
            self._mission_id = None

        self._fsm.reset()
        return result

    # ── Topic callbacks ───────────────────────────────────────────────────────

    def _cb_target(self, msg: TerrainTarget):
        self._latest_target = msg
        # Update goal from 3-D waypoint if available
        if msg.target_found and msg.waypoint.header.frame_id:
            new_x = msg.waypoint.point.x
            new_y = msg.waypoint.point.y
            # Only update D* Lite goal when the waypoint changes meaningfully
            if (math.isnan(self._goal_x)
                    or math.hypot(new_x - self._goal_x, new_y - self._goal_y) > 0.3):
                self._goal_x = new_x
                self._goal_y = new_y
                if self._use_dstar:
                    self._dstar.set_goal(new_x, new_y)

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
                # D* Lite: get collision-free intermediate waypoint
                wp_x, wp_y = self._goal_x, self._goal_y
                if self._use_dstar:
                    wp = self._dstar.next_waypoint(
                        self._rover_x, self._rover_y, lookahead_m=2.0
                    )
                    if wp is not None:
                        wp_x, wp_y = wp
                    # If D* Lite has no path, fall back to direct waypoint
                self._planned_wp_x, self._planned_wp_y = wp_x, wp_y

                out = self._controller.compute(
                    self._rover_x, self._rover_y, self._rover_yaw,
                    wp_x, wp_y,
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
