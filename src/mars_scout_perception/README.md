## `mars_scout_perception`

Perception package placeholder.

Responsibilities:

- Subscribe to RGB (and optional depth thumbnail)
- Convert (image, prompt) into a grounded 2D hypothesis:
  - ROI + center point
  - confidence score
  - model id

Planned baselines:

- B1: CLIP patch-grid scoring (cheap, reproducible)
- B2: VLM grounding (primary)

