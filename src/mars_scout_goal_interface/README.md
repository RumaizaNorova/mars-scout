## `mars_scout_goal_interface`

Goal interface package placeholder.

Responsibilities:

- Accept a natural language goal
- Publish a structured `GoalCommand` for the rest of the stack

Initial plan:

- Topic-based interface (`std_msgs/String` or custom msg) for simplest integration
- Upgrade to a service/action later if needed

