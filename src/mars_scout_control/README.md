## `mars_scout_control`

Control package placeholder.

Responsibilities:

- Drive from current pose to a waypoint
- Publish `/cmd_vel`
- Apply safety constraints:
  - max linear/angular velocity
  - timeout
  - stuck detector
  - optional simple obstacle stop using depth

