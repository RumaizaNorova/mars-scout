# Full rover state snapshot published by the sim bridge.

std_msgs/Header header

# Pose in map frame
geometry_msgs/PoseWithCovariance pose

# Velocity in base_link frame
geometry_msgs/TwistWithCovariance velocity

# Agent finite-state-machine state
string fsm_state          # SEARCHING | APPROACHING | ARRIVED | ABORTED

# Active query (empty if none)
string active_query_id
string active_query_text

# Distance to current waypoint (metres); NaN if no active waypoint
float32 distance_to_goal

# How long the current query has been running (seconds)
float32 query_elapsed_sec
