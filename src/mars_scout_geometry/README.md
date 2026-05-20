## `mars_scout_geometry`

Geometry package placeholder.

Responsibilities:

- Convert a 2D target hypothesis (pixel u,v or ROI) into a 3D waypoint:
  - Backproject using depth + camera intrinsics
  - Transform into the chosen fixed frame using tf2
  - Enforce a “ground” constraint (project to plane) for MVP

