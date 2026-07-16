# ARES Project Handbook
## Everything You Need to Know — From Zero

*A complete reference for anyone working on the Autonomous Rover Exploration System.*
*No background assumed. Read in order the first time, use as reference after that.*

---

## Table of Contents

1. [The Big Picture — What Are We Building and Why?](#1-the-big-picture)
2. [Mars — The Planet](#2-mars-the-planet)
3. [Jezero Crater — Why Here?](#3-jezero-crater)
4. [Real Mars Rovers — What They Do](#4-real-mars-rovers)
5. [The Perseverance Rover in Detail](#5-perseverance-rover)
6. [The Rocker-Bogie Suspension](#6-rocker-bogie-suspension)
7. [NASA's AEGIS — The System We're Comparing Against](#7-aegis)
8. [Simulation with Isaac Sim](#8-isaac-sim)
9. [USD — How 3D Worlds Are Described](#9-usd)
10. [Physics Simulation — PhysX](#10-physx)
11. [ROS2 — The Nervous System](#11-ros2)
12. [The Camera and Depth Sensing](#12-cameras-and-depth)
13. [Vision-Language Models — The AI Brain](#13-vlm)
14. [Navigation — How the Rover Finds Its Way](#14-navigation)
15. [The Full ARES Pipeline — How Everything Connects](#15-ares-pipeline)
16. [The Science — What We're Looking For and Why](#16-the-science)
17. [HiRISE — Real NASA Terrain Data](#17-hirise)
18. [The Research Paper — What We're Claiming](#18-the-paper)
19. [Glossary](#19-glossary)

---

## 1. The Big Picture

### What is ARES?

ARES stands for **Autonomous Rover Exploration System**. It is a complete Mars rover simulation where a rover drives itself across real Mars terrain, looks at rocks with an AI camera, decides which rocks are scientifically interesting, drives to them, and logs what it found — all without a human telling it what to do.

### Why does this matter?

Right now, real Mars rovers like Perseverance operate like this:

```
Day 1 (Earth): Scientists look at yesterday's photos → decide where to go
Day 1 (Mars):  Rover receives commands from Earth → drives → takes photos
Day 2 (Earth): Scientists look at today's photos → decide next move
...
```

The problem: **radio signals between Earth and Mars take 3 to 22 minutes each way** depending on orbital positions. A full command-receive-execute-report cycle can take 2 full Earth days. For a rover moving at 0.14 m/s, this is incredibly slow.

The solution ARES explores: **put the intelligence on the rover itself**. Let it decide what's interesting, drive there autonomously, and only report back the important findings. This is called *onboard autonomous science targeting*.

### What ARES does step by step:

1. Rover drives across 500×500m of real Jezero Crater terrain
2. Camera captures what's ahead
3. AI (Vision-Language Model) looks at the image and asks: *"Is there a scientifically interesting rock here?"*
4. If yes: navigate to it, photograph it from all angles, log the finding
5. If no: keep driving, scan more terrain
6. Build a map of everything found during the mission

### Why is it a research contribution?

No existing open-source system combines all of:
- Real NASA elevation data as terrain
- Physically accurate rover suspension
- VLM-based geological targeting
- Quantitative comparison against a real baseline (AEGIS)

ARES is the first to put all four together in one system.

---

## 2. Mars — The Planet

### Basic facts you need to know

| Property | Mars | Earth | Why it matters for ARES |
|---|---|---|---|
| Gravity | 3.72 m/s² | 9.81 m/s² | Wheels behave differently; physics must match |
| Day length | 24h 39min 35s (1 sol) | 24h 00min | Mission planning in "sols" not days |
| Distance from Earth | 54–401 million km | — | 3–22 minute signal delay each way |
| Surface temperature | -80°C to +20°C | -90°C to +60°C | Why electronics are heated |
| Atmosphere | 0.6% of Earth's | 100% | No weather effects in sim (simplified) |
| Surface composition | Iron oxide (rust) | Varied | Explains the red color and our material choices |

### Why does Mars look red?

The surface is covered in iron oxide dust — essentially rust. When iron-rich rocks were exposed to trace amounts of water and oxygen billions of years ago, they rusted. The dust is so fine it covers everything and even colors the sky pink/orange. This is why our simulation uses warm amber/rust PBR materials.

### What is a sol?

A **sol** is one Martian day = 24 hours 37 minutes. Mars rover missions are measured in sols rather than Earth days. Perseverance landed on Sol 0. Every science decision, drive plan, and photo session is scheduled by sol. ARES's "sol planning" mode simulates a rover completing a science survey within a single Martian day.

### The Martian sky in our simulation

We render the sky as a hazy pink-peach dome (DomeLight at 600 lux intensity, pink-peach color). The sun is a DistantLight at 3500K (amber) at 28° elevation angle — calibrated to Jezero Crater morning conditions. This is not aesthetic choice — it affects how rocks look in camera images, which affects how the VLM interprets them.

---

## 3. Jezero Crater

### What is it?

Jezero Crater is a **49 km wide impact crater** in the northern hemisphere of Mars at 18.4°N, 77.7°E. It was formed by a meteor impact approximately 3.9 billion years ago.

### Why was it chosen for Perseverance?

Evidence of an **ancient lake**. About 3.5 billion years ago, water flowed into Jezero through two river valleys (visible from orbit), filled the crater, and stayed for potentially millions of years. A delta formed where the river met the lake — exactly like river deltas on Earth. The crater then dried out.

Why does an ancient lake matter? **Where there was water, there may have been life**. Microbial life on Mars (if it ever existed) would have lived in or near water. The lake sediments in Jezero are one of the best places to look for biosignatures — chemical signatures of ancient life. This is why Perseverance was sent there.

### The terrain in ARES

We use a **500×500 metre patch** from the centre of the crater floor:

- **Elevation range**: -2454m to -2430m (Mars reference datum) — 24.3m variation across our patch
- **Resolution**: 1 metre per pixel (each grid point is 1m apart)
- **Terrain source**: HiRISE DTM (Digital Terrain Model) — explained in Section 17
- **What it looks like**: Gently undulating flat floor with occasional ridges and rocky outcrops

The crater floor is geologically fascinating: ancient lake bed sediments, volcanic rocks that predate the lake, and float rocks (rocks that moved from elsewhere). Our simulation places rocks following the actual statistical distribution measured at Mars landing sites.

---

## 4. Real Mars Rovers — What They Do

### The history (briefly)

| Rover | Year | Duration | Notable achievement |
|---|---|---|---|
| Sojourner | 1997 | 83 sols | First Mars rover. Size of a microwave. |
| Spirit | 2004 | 2208 sols | Found evidence of past water exposure |
| Opportunity | 2004 | 5111 sols | Drove 45.16 km — marathon on Mars |
| Curiosity | 2012 | Still operating | Found organic molecules, measured radiation |
| Perseverance | 2021 | Still operating | Searching for biosignatures, caching samples |

### What a rover actually does each sol

1. **Wake up** — heaters turn on electronics
2. **Receive uplink** — commands from Earth arrive (sent the night before Earth time)
3. **Execute drive** — autonomous driving along commanded waypoints
4. **Science stops** — at commanded locations, take photos, use instruments
5. **Data downlink** — transmit all data via Mars orbiters to Earth
6. **Sleep** — conserve battery overnight

The bottleneck is step 2 — scientists must **plan the entire next sol** based on photos from the current sol, all while accounting for the 20+ minute signal delay. ARES's autonomous targeting removes this bottleneck for the science-selection step.

---

## 5. Perseverance Rover

### Why we model after Perseverance

Perseverance is the most advanced Mars rover ever built and the one currently operating at Jezero Crater — exactly where our terrain data comes from. Comparing our simulation against Perseverance's actual decisions is the most meaningful benchmark possible.

### Perseverance specifications

| Parameter | Value | ARES implementation |
|---|---|---|
| Mass | 1025 kg | ~970 kg (rover_builder approximation) |
| Wheel diameter | 52.5 cm | 52.5 cm (exact match) |
| Track width | 2.78 m | 2.78 m (exact match) |
| Wheelbase | 2.83 m | 2.83 m (exact match) |
| Ground clearance | 0.60 m | 0.60 m (exact match) |
| Max speed | 0.14 m/s | 0.5 m/s (sim, faster for testing) |
| Wheels | 6 × titanium spokes | 6 × simulated cylinders |
| Suspension | Rocker-bogie | Full rocker-bogie with PhysX joints |

### Perseverance's cameras (real)

| Camera | Purpose | ARES equivalent |
|---|---|---|
| MastCam-Z | Science imaging, stereo | MastCamL + MastCamR |
| Hazard Avoidance Cameras (HazCam) | Obstacle detection | Planned Stage 3 |
| Navigation Cameras (NavCam) | Path planning | Same as MastCam in ARES |
| WATSON | Close-up rock photos | Not modelled |
| RIMFAX | Ground-penetrating radar | Not modelled |

### Perseverance's science instruments (real)

- **PIXL** — X-ray spectrometer to detect chemical elements in rocks
- **SHERLOC** — Raman spectrometer to detect organic compounds
- **SuperCam** — Laser that vaporises rock and analyses the plasma
- **MOXIE** — Produces oxygen from CO₂ (technology demo)

ARES doesn't simulate the instruments — instead the VLM acts as a visual surrogate that classifies rocks by appearance, which is analogous to what SuperCam and PIXL do chemically.

---

## 6. The Rocker-Bogie Suspension

This is one of the most important mechanical concepts in the project. Understanding it helps you understand why our rover physics are complex.

### The problem: crossing rocks

A wheeled vehicle on Earth struggles with rocks larger than its wheel radius. The wheel hits the rock face and stops. Most wheels are designed for smooth roads.

Rover wheels on Mars need to climb over rocks **taller than half the wheel diameter**. On smooth terrain, this doesn't matter. But Mars is covered in rocks — some the size of boulders.

### The solution: rocker-bogie

Imagine you're pushing a shopping cart with a bad wheel. The cart has 4 fixed wheels and tips up when one wheel hits a curb. Now imagine the cart had jointed legs that could flex — even if one wheel goes up, the others stay on the ground.

That's the rocker-bogie concept, but more sophisticated:

```
             CHASSIS
           /         \
    ROCKER-L         ROCKER-R
    /      \          /      \
WHEEL-FL  BOGIE-L  BOGIE-R  WHEEL-FR
           /  \      /  \
       WHL-ML WHL-RL WHL-MR WHL-RR
```

**How it works:**
- Each side has a **rocker** — a rigid beam that pivots where it connects to the chassis
- At the rear end of each rocker is a **bogie** — another beam that pivots
- The bogie carries the middle and rear wheels
- The front wheel attaches directly to the rocker

When the rover encounters a rock:
- The front wheel rides up the rock
- The rocker tilts, dropping the rear (bogie) end
- The bogie compensates, keeping all three wheels on ground
- The chassis tilts only slightly (average of both rockers)

**The differential bar:**
A horizontal bar connecting the left and right rockers across the chassis top. When the left side goes over a rock and tilts, the bar tilts the right side slightly to compensate, keeping the chassis level. This is what makes the 6-wheel system capable of 45° tilts without tipping.

**The kinematic loop problem:**
The differential bar creates a **closed kinematic loop** — a chain of rigid links that forms a loop. Physics engines can't simulate loops directly (they'd over-constrain the system). 

Our solution: **break the loop by making one side's differential connection passive** — zero stiffness, just damping. The physics still work, the rover still levels, but there's no loop for PhysX to struggle with.

### Why ARES builds this from scratch

Every joint, every mass, every limit is physically meaningful. This matters for the paper: we can say exactly what physics our rover obeys and why it behaves realistically on uneven terrain. A pre-made URDF robot would hide all this.

---

## 7. AEGIS — The System We're Comparing Against

### What is AEGIS?

AEGIS (Autonomous Exploration for Gathering Increased Science) is NASA JPL's **real onboard autonomous science system**, first deployed on the Opportunity rover in 2010–2011 and later on Curiosity. It is the baseline we compare ARES against.

### How AEGIS works

1. Rover arrives at a new location
2. NavCam takes a panoramic image
3. AEGIS runs a classifier on the image (looking for rock targets)
4. If a target is found: autonomously positions the ChemCam laser and fires
5. Reports the spectrometer data in the next downlink

AEGIS uses **traditional computer vision** — edge detection, texture analysis, colour thresholding. It was state-of-the-art in 2008. It does not use deep learning.

### AEGIS performance (from published papers)

- **Precision**: ~70–80% range (exact number: verify from Wagstaff et al. papers before citing)
- **Recall**: ~65–75% range (exact number: verify from Wagstaff et al. papers before citing)
- **Main weakness**: misses rocks covered in dust, fails on novel rock types

> ⚠️ **Paper note**: Do not use the above numbers directly. Look up the actual values in:
> Wagstaff et al., "Guiding Scientific Discovery with Explanations Using DEMUD" (AAAI 2013)
> and the AEGIS deployment papers on Opportunity/Curiosity before submission.

### Why ARES should beat AEGIS

The VLM (Moondream2) was trained on billions of images including geological content. It understands texture, shape, colour relationships, and context in ways classical computer vision cannot. Asking it *"is this rock scientifically interesting?"* is a fundamentally different approach.

The comparison in our paper:
- Run ARES on a mission
- Run a simulated AEGIS (re-implemented as a classical classifier) on the same mission
- Compare precision, recall, and science value of selected targets
- Quantify improvement

---

## 8. Isaac Sim — The Simulator

### What is Isaac Sim?

Isaac Sim is NVIDIA's physics-accurate robotics simulator. It runs on a GPU and renders photorealistic 3D scenes while simultaneously simulating physics (gravity, friction, joint forces, collisions). Think of it as a video game engine specifically designed for robotics research.

### Why Isaac Sim instead of Gazebo or other simulators?

| Feature | Isaac Sim | Gazebo (common alternative) |
|---|---|---|
| Rendering | Photorealistic RTX ray tracing | Basic OpenGL |
| Physics | NVIDIA PhysX (GPU-accelerated) | ODE/Bullet (CPU) |
| Camera sim | RTX-accurate depth, lens effects | Simplified |
| VLM compatibility | Photorealistic images work with VLMs | Synthetic images don't transfer |
| Setup complexity | Higher | Lower |

The last point is crucial: **our VLM was trained on real photographs**. If the simulated camera produces unrealistic images, the VLM won't work correctly. Isaac Sim's ray-traced rendering produces images close enough to real Mars photos that the VLM can interpret them meaningfully.

### How Isaac Sim runs in ARES

1. `setup_scene.py` starts Isaac Sim as a Python process
2. Loads the terrain and rover into USD stage
3. Runs the physics simulation loop
4. Each frame: publishes camera images + odometry to ROS2 topics
5. Reads velocity commands from ROS2 and applies to rover wheels

### Headless vs GUI mode

**Headless**: No display. Runs as a pure computation process. You can't see anything visually but it's faster and works on a server without a screen (like Vast.ai without VNC).

**GUI mode** (with VNC): Full 3D viewport where you can orbit the camera, pause physics, inspect joints, and watch the rover drive in real time. Like a video game you can interact with.

In ARES: `python3 isaac_sim/setup_scene.py` = headless, `--headed` = GUI via VNC.

---

## 9. USD — Universal Scene Description

### What is USD?

USD (Universal Scene Description) was created by Pixar for animated films. Every object in a 3D scene — its shape, position, colour, physics properties — is described as text in a `.usd` file. NVIDIA adopted USD as the foundation of Isaac Sim.

### Why USD matters for ARES

We build the entire rover and terrain **programmatically in Python** using USD APIs. This means:
- No external asset files needed
- Every dimension is exactly what we set in code
- We can inspect, debug, or modify any property at runtime
- The paper can cite exact numerical values for everything

### USD concepts you need to know

**Prim**: The basic unit of a USD scene. Like a "thing" — could be a mesh, a camera, a light, or just an empty transform. Every prim has a path like `/World/Rover/Body`.

**Xform**: A prim that holds a 3D transformation (position, rotation, scale). Like a folder in a file system — organises other prims underneath it.

**Mesh**: A prim that defines a 3D shape using vertices and faces (triangles). Our wheels are cylinder meshes. Our rocks are icosphere meshes.

**Stage**: The entire scene — all prims, all relationships. Like the "document" of the USD world.

**API**: A set of behaviours applied to a prim. `UsdPhysics.RigidBodyAPI` makes a mesh simulate physics. `UsdPhysics.DriveAPI` makes a joint act like a motor.

### How ARES builds the rover in USD

```python
# Define a box mesh for the chassis
mesh = UsdGeom.Mesh.Define(stage, "/World/Rover/Body")
mesh.CreatePointsAttr(...)     # 8 vertices
mesh.CreateFaceVertexIndicesAttr(...)  # 12 triangles

# Apply physics
UsdPhysics.RigidBodyAPI.Apply(prim)   # "this mesh obeys gravity"
UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(800.0)  # 800 kg

# Apply a motor
UsdPhysics.DriveAPI.Apply(joint, "Y")  # joint can be driven around Y axis
drive.CreateDampingAttr(500.0)         # motor torque
drive.GetTargetVelocityAttr().Set(5.0) # spin at 5 rad/s
```

Every property of every object in the simulation is controlled this way.

---

## 10. PhysX — The Physics Engine

### What is PhysX?

PhysX is NVIDIA's physics simulation library. It runs on the GPU and computes:
- Gravity and rigid body motion
- Collisions (when does wheel touch terrain?)
- Joint forces (how does the rocker-bogie flex under load?)
- Friction (how much do wheels slip on Mars regolith?)

### Key physics concepts in ARES

**Rigid body**: An object that doesn't deform when forces act on it. Every rover part (chassis, wheels, rockers) is a rigid body. They can move and rotate but not bend.

**Articulation**: A chain of rigid bodies connected by joints. The rover is one articulation: chassis → rocker → bogie → wheels, all connected. PhysX solves the forces in the whole chain simultaneously (much more stable than solving each joint independently).

**Revolute joint**: A joint that allows rotation around one axis only — like a hinge. Our wheel joints are revolute (spin around Y axis). Our rocker joints are also revolute (rock around Y axis, limited to ±30°).

**DriveAPI**: Turns a passive revolute joint into a motor. You set a target velocity; the drive applies torque to reach that velocity. Our wheel drives have:
- `stiffness = 0` (not a position controller — doesn't try to hold a specific angle)
- `damping = 500 N·m` (torque applied proportional to velocity error)
- `targetVelocity = X rad/s` (what we set from rover_controller.py)

**Friction and Mars regolith**: Mars surface is loose sandy soil (regolith) with rocks. Friction between wheel and surface determines if wheels slip. We use PhysX's default friction which approximates this reasonably.

**Why Mars gravity matters**: At 3.72 m/s² (vs Earth's 9.81 m/s²), the rover weighs less. This means:
- Less normal force on wheels → less friction → more tendency to slip
- Falls happen in slow motion (rover dropping from 30m takes 4 seconds)
- Driving over rocks requires different torques

---

## 11. ROS2 — The Nervous System

### What is ROS2?

ROS2 (Robot Operating System 2) is a framework for building robot software. It's not an operating system — it's a **messaging system** that lets different programs talk to each other. Think of it as a bus where programs can send and receive data.

In ARES: Isaac Sim publishes camera images on this bus. The perception AI reads those images from the bus. The navigation system publishes velocity commands on the bus. Isaac Sim reads those commands and drives the wheels. None of these programs need to know about each other directly.

### Core concepts

**Node**: A program that participates in ROS2. Examples in ARES:
- `sim_bridge_node` — translates Isaac Sim topics to ARES standard topics
- `perception_node` — runs the VLM
- `projection_node` — converts image detections to 3D coordinates
- `agent_node` — makes navigation decisions

**Topic**: A named channel on the message bus. Examples:
- `/rover/camera/image_raw` — the rover's camera feed
- `/rover/odom` — where the rover is and how fast it's moving
- `/rover/cmd_vel` — velocity command (go forward, turn left)

**Publisher**: A node that sends messages onto a topic.
**Subscriber**: A node that reads messages from a topic.

**Message type**: The format of data on a topic. Examples:
- `sensor_msgs/Image` — a camera frame (width, height, pixel data)
- `nav_msgs/Odometry` — position + velocity + covariance
- `geometry_msgs/Twist` — linear and angular velocity commands

### The topics in ARES

| Topic | Direction | Message type | What it carries |
|---|---|---|---|
| `/rover/camera/image_raw` | Isaac Sim → Perception | Image | RGB camera frame from MastCam |
| `/rover/camera/depth/image_raw` | Isaac Sim → Projection | Image | Depth map (metres per pixel) |
| `/rover/camera/camera_info` | Isaac Sim → Projection | CameraInfo | Focal length, image size (for 3D math) |
| `/rover/odom` | Isaac Sim → Agent | Odometry | Rover position + velocity in world |
| `/rover/state` | Bridge → Dashboard | RoverState | Full rover status (custom message) |
| `/rover/cmd_vel` | Agent → Isaac Sim | Twist | Drive command: forward speed + turn rate |
| `/rover/chase_cam` | Isaac Sim → Dashboard | Image | 3rd-person camera view |

### QoS — Quality of Service

When you subscribe to a topic, you can choose how reliable the delivery is:

**RELIABLE**: Every message is guaranteed to arrive. If it gets dropped, it's resent. Good for commands and state.

**BEST_EFFORT**: Messages may be dropped. Good for sensor data where losing one frame doesn't matter and low latency is more important.

**Critical rule** (we learned this the hard way): A BEST_EFFORT publisher and a RELIABLE subscriber are **incompatible** — the subscriber receives nothing. We fixed this by making all ARES publishers RELIABLE.

### TF2 — Transform Tree

TF2 (transform library) maintains a tree of coordinate frames. Every sensor, body part, and reference frame has a position relative to its parent.

In ARES:
```
map
└── odom
    └── base_link  (rover chassis)
        └── camera_optical_frame  (MastCam)
```

When the projection node needs to know "where in 3D world space is this pixel?", it looks up the transform chain to find where the camera is in the world, then does the math.

---

## 12. Cameras and Depth Sensing

### How the MastCam works in simulation

Isaac Sim renders the scene from the camera's perspective. The `ROS2CameraHelper` OmniGraph node captures that rendered frame and publishes it as a `sensor_msgs/Image` message.

Two types of output:
- **RGB** (`type="rgb"`): Colour image, like a photo
- **Depth** (`type="depth"`): Each pixel's value = distance to that point in metres. A pixel showing a rock 3 metres away has value 3.0. This is impossible to get from a normal photo.

### Camera intrinsics — the CameraInfo message

To convert a depth image into 3D coordinates, you need to know the camera's geometry. The `CameraInfo` message carries:

**Focal length (fx, fy)**: How "zoomed in" the camera is. Higher focal length = narrower field of view. For ARES: `fx = fy = 640` pixels (derived from 18mm focal length on a standard sensor).

**Principal point (cx, cy)**: Where the optical axis intersects the image plane. Usually the image centre. For ARES: `cx = 640, cy = 360` (centre of 1280×720 image).

**Projection formula**: For a 3D point (X, Y, Z) in camera coordinates:
```
pixel_x = fx * (X/Z) + cx
pixel_y = fy * (Y/Z) + cy
```

To go backwards (pixel + depth → 3D point):
```
X = (pixel_x - cx) * depth / fx
Y = (pixel_y - cy) * depth / fy
Z = depth
```

This is what `projection_node` does: takes a bounding box from the VLM, finds the pixels inside it, looks up their depth values, converts to 3D, and produces a waypoint for the rover to navigate to.

### Why Isaac Sim doesn't publish CameraInfo

Isaac Sim's `ROS2CameraHelper` publishes RGB and Depth images but **not** CameraInfo. This was a real problem we hit and solved by fabricating the CameraInfo in `sim_bridge_node.py`:

```python
# We know the camera: 18mm focal length, 36mm sensor, 1280×720 image
fx = (18 / 36) * 1280 = 640.0
fy = 640.0
cx = 1280 / 2 = 640.0
cy = 720  / 2 = 360.0
```

This fabricated message is published at 10Hz so projection_node always has valid intrinsics.

---

## 13. Vision-Language Models — The AI Brain

### What is a VLM?

A Vision-Language Model (VLM) is an AI that understands both images and text. You can show it a photo and ask a question in plain English, and it answers in plain English.

```
Input:  [image of rocks] + "Are there any geologically interesting rock formations here?"
Output: "Yes, there appears to be a layered sedimentary outcrop in the left portion 
         of the image, with distinct horizontal stratification suggesting water 
         deposition."
```

This is fundamentally different from traditional computer vision which classifies images into fixed categories. A VLM reasons about what it sees.

### Moondream2 — what we use

**Moondream2** is a small, efficient VLM created specifically to run on limited hardware. 

| Property | Value |
|---|---|
| Model size | ~1.9 billion parameters |
| VRAM at 4-bit quantization | ~2–3 GB (re-measure at Stage 4) |
| Speed on RTX 3090 | re-benchmark at Stage 4 implementation |
| Task | Visual question answering + detection |

"4-bit quantization" means the model weights are stored with lower precision (4 bits per number instead of 32). This reduces quality slightly but cuts memory usage by 8×, making the model fit alongside Isaac Sim on the same GPU.

### What ARES asks Moondream2

Geological query templates:
- `"Are there any unusual rock formations, layering, or geological features of interest in this image?"`
- `"Identify any rocks that appear different in texture, colour, or shape from the surrounding terrain."`
- `"Is there evidence of sedimentary layering, erosion, or mineral deposits visible here?"`

The VLM also does **object detection**: returns bounding boxes around specific targets.
```
Query: "Find rocks that appear geologically interesting"
Output: [{"label": "layered outcrop", "bbox": [245, 180, 420, 310], "score": 0.87}]
```

### Why this beats classical computer vision for science targeting

Classical approach (AEGIS): "Find pixels with high edge density and unusual colour" → brittle, fails on dust-covered rocks, misses novel formations.

VLM approach: "Does this look scientifically interesting?" → leverages understanding of geology, context, visual reasoning trained from millions of images.

---

## 14. Navigation — How the Rover Finds Its Way

### The navigation problem

The rover has a camera image with an interesting rock identified. The rock might be 50 metres away, with other rocks and uneven terrain between them. The rover needs to plan a path that:
- Reaches the science target
- Avoids driving over rocks it can't cross
- Updates the plan as new obstacles are discovered

### Traversability grid

Before planning a path, the rover computes a **traversability grid** — a top-down grid map where each cell is rated: safe to drive (0.0) to impossible (1.0).

How traversability is computed from depth images:
- High slope between adjacent points = risky
- Height above baseline > rover clearance = obstacle
- Unknown (no depth data) = unknown risk

Each cell in the grid covers 0.25m × 0.25m. For our 500×500m terrain: 2000×2000 grid cells.

### D* Lite path planning

**D\* Lite** (Dynamic A* Lite) is the path planning algorithm we use. It's the same algorithm used by real Mars rovers.

Why D* Lite over simpler algorithms (like A*)?
- **Dynamic replanning**: Mars terrain isn't fully known in advance. As the rover drives, its HazCam sees new obstacles. D* Lite can update the path efficiently when new obstacles are discovered — much faster than replanning from scratch.
- **Incremental**: Only recomputes the parts of the plan that changed.
- **Optimal**: Finds the shortest safe path.

How it works (simplified):
1. Start with a rough traversability map (mostly unknown)
2. Plan initial path from rover position to science target
3. Drive forward
4. HazCam reveals new detail → update traversability grid
5. D* Lite detects which plan segments changed → efficiently updates only those
6. Repeat until target reached or declared unreachable

### Odometry — knowing where you are

**Odometry** is dead-reckoning navigation: track how far wheels have turned and in which direction, accumulate the estimates, derive position.

```
rover moved forward 1.0m + turned right 15° → update position estimate
```

Problem: wheels slip on sand. Each slip = position error that accumulates. On real Mars, visual odometry (tracking features in images) corrects this. In ARES, Isaac Sim provides ground-truth odometry directly (no wheel slip error) since we're not studying localisation.

---

## 15. The Full ARES Pipeline

Here is exactly how data flows from camera to wheel movement:

```
ISAAC SIM
│
│  Physics step → camera renders frame
│
├─► /isaac_camera/rgb   ─────────────────────────────────┐
├─► /isaac_camera/depth ──────────────────────────────────┤
├─► /isaac_camera/chase (3rd person view for dashboard)   │
└─► /odom               ──────────────────────────────┐   │
                                                       │   │
SIM BRIDGE NODE                                        │   │
│  Remaps Isaac Sim topics → ARES standard topics      │   │
│  Fixes zero timestamps                               │   │
│  Fabricates CameraInfo at 10Hz                       │   │
│  Publishes TF tree (map→odom→base_link→camera)       │   │
│                                                      │   │
├─► /rover/camera/image_raw  ◄───────────────────────────┘
├─► /rover/camera/depth/image_raw ◄───────────────────────┘
├─► /rover/camera/camera_info (fabricated)
├─► /rover/odom ◄─────────────────────────────────────────┘
└─► /rover/state (derived: pose + fsm_state + query text)

PERCEPTION NODE
│  Subscribes: /rover/camera/image_raw
│  Runs Moondream2 VLM on each frame
│  Query: "find geologically interesting rocks"
│  Output: bounding boxes + confidence scores
│
└─► /rover/detections

PROJECTION NODE
│  Subscribes: /rover/detections + /rover/camera/depth + /rover/camera/camera_info
│  For each detection bounding box:
│    → find median depth in bounding box
│    → convert (pixel_x, pixel_y, depth) → 3D point using intrinsics
│    → transform 3D point from camera frame to world frame (via TF)
│  Output: 3D waypoint in world frame
│
└─► /rover/science_waypoint

AGENT NODE (FSM)
│  States: SEARCHING → APPROACHING → INVESTIGATING → RETURNING
│  Subscribes: /rover/odom + /rover/science_waypoint
│  Runs D* Lite on traversability grid
│  Publishes velocity commands
│
└─► /rover/cmd_vel ──────────────────────────────────────────┐
                                                              │
SIM BRIDGE NODE                                               │
│  Forwards /rover/cmd_vel → /cmd_vel (Isaac Sim format)      │
└─► /cmd_vel ◄────────────────────────────────────────────────┘
                                                              │
ISAAC SIM (rover_controller.py)                               │
  Reads /rover/cmd_vel ◄───────────────────────────────────────┘
  Differential drive: linear, angular → 6 wheel velocities
  Sets DriveAPI target velocities on all 6 wheels
  Physics step executes wheel motion
  → back to top of loop
```

The complete loop runs at ~15 Hz (limited by camera publishing rate).

---

## 16. The Science — What We're Looking For

### Geological targets on Mars

Not all rocks are equal scientifically. What makes a rock worth investigating:

**Sedimentary layering**: Horizontal bands of different colour/texture in rock. Indicates the rock was deposited layer by layer under water (like lake bed sediment). These layers can trap biosignatures (chemical evidence of past life).

**Colour anomalies**: A rock that's different colour from surroundings might have a different mineral composition. Purple or grey rocks amid red iron oxide could indicate sulphate minerals (associated with water chemistry).

**Unusual texture**: Smooth rounded rocks were likely tumbled in water. Pitted surfaces might indicate gas bubble escape during volcanic solidification. Crystalline surfaces indicate slow cooling.

**Veins**: White or light-coloured lines running through rock. On Earth, veins form when water flows through cracks and deposits minerals. On Mars, veins indicate past water presence.

**Float rocks**: Rocks sitting on the surface that don't match surrounding geology — they were carried there by ancient water flow or impact ejecta.

### The CFA formula — how we place rocks

The **Cumulative Fractional Area (CFA)** formula from Golombek et al. 2008 describes the statistical distribution of rock sizes on Mars:

```
q(k) = 1.79 × exp(-0.152/k)
```

Where:
- `k` = rock diameter in metres
- `q(k)` = fraction of surface covered by rocks larger than k

This formula was derived by measuring rocks at every Mars landing site. It means:
- Smaller rocks are exponentially more common than larger ones
- A 2.5m boulder is rare; a 0.25m pebble is common
- The distribution is consistent across different Mars locations

ARES uses this formula to determine how many rocks of each size class to place in our 500×500m terrain, making the rock distribution scientifically realistic.

### Perseverance mission ground truth

Perseverance publishes daily sol reports including:
- GPS-equivalent coordinates of each science stop
- What instruments were used
- What was found (chemical composition, mineral identification)
- Science team's assessment of each target

We can download this data and compare: at a given location, did our VLM identify the same targets that Perseverance's science team selected? This is the ground truth comparison in our paper.

---

## 17. HiRISE — Real NASA Terrain Data

### What is HiRISE?

HiRISE (High Resolution Imaging Science Experiment) is a camera on the Mars Reconnaissance Orbiter (MRO) spacecraft. It has been orbiting Mars since 2006, photographing the surface at 25–50 cm/pixel — the highest resolution camera ever sent to another planet.

From these photos, scientists compute **DTMs (Digital Terrain Models)** — elevation maps where each pixel represents the height of that point on the Mars surface.

### The file we downloaded

```
JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif
```

Breaking this down:
- `JEZ` — Jezero Crater
- `hirise_soc_006` — HiRISE SOC (Science Operations Center) processing, version 6
- `DTM` — Digital Terrain Model (elevation data)
- `MOLAtopography_DeltaGeoid` — elevation referenced to MOLA (Mars Orbiter Laser Altimeter) areoid (Mars equivalent of sea level)
- `1m` — 1 metre per pixel resolution
- `Eqc` — Equirectangular projection

**File size**: 1.76 GB  
**Coverage**: ~55 km × 52 km of Jezero Crater  
**Our usage**: 500×500 pixel crop = 500m × 500m at 1m/pixel

### What we do with it

```python
# Load GeoTIFF with rasterio
with rasterio.open("jezero_hirise.tif") as src:
    raw = src.read(1)  # elevation values in metres

# Crop 500×500m patch from centre
elevation = raw[cy-250:cy+250, cx-250:cx+250]

# Subtract minimum so terrain starts at z=0
elevation -= elevation.min()

# Result: 500×500 array of float32 elevation values
# Each value = how many metres above the lowest point in our patch
```

The resulting elevation range in our simulation: 0m to ~24.3m.

### MOLA — the reference datum

MOLA (Mars Orbiter Laser Altimeter) scanned the entire Mars surface with a laser altimeter, producing a global elevation map. All HiRISE elevations are referenced to the MOLA areoid — the Mars equivalent of sea level. This is why our terrain values are around -2450m: Jezero Crater floor is about 2.45 kilometres below the Mars reference datum.

---

## 18. The Research Paper

### What we're claiming

The paper makes three primary claims:

1. **"We built a physically accurate simulation of a Perseverance-class rover on real Jezero Crater terrain"**
   - Evidence: exact wheel/wheelbase/mass specs, 1m/pixel real HiRISE data, Mars gravity

2. **"A VLM (Moondream2) can perform autonomous geological science targeting in simulation"**
   - Evidence: detection precision/recall on science target categories

3. **"This VLM-based approach outperforms AEGIS (NASA's onboard targeting system) in simulation"**
   - Evidence: quantitative comparison on same terrain, same targets

### What kind of paper is it?

**Systems paper** at a robotics conference. It presents a complete system — not a new algorithm, not new theory — but a new integration that produces a new capability. The contribution is the whole pipeline.

Target conferences:
- **ICRA** (International Conference on Robotics and Automation) — the top robotics conference
- **IROS** (Intelligent Robots and Systems) — the other top robotics conference

Both have a **Field Robotics** or **Space Robotics** track where this work fits perfectly.

### Limitations to acknowledge

- Simulated terrain ≠ real Mars (no dust, no thermal effects, perfect sensor data)
- VLM was not fine-tuned on Mars geology specifically
- No multi-sol accumulation yet (single mission run)
- No real hardware deployment

Acknowledging limitations honestly is a sign of good science. Reviewers trust papers more when authors know what their work doesn't do.

---

## 19. Glossary

| Term | Definition |
|---|---|
| **Areoid** | The Mars equivalent of sea level — a reference elevation surface |
| **Articulation** | A chain of rigid bodies connected by joints, simulated together |
| **Autonomy** | System's ability to make decisions and act without human input |
| **Biosignature** | Chemical or physical evidence of past or present life |
| **CFA** | Cumulative Fractional Area — Mars rock size-frequency distribution formula |
| **ChemCam / SuperCam** | Laser instruments on Curiosity / Perseverance that vaporize rock for analysis |
| **cmd_vel** | ROS2 convention for velocity commands: linear.x (m/s) + angular.z (rad/s) |
| **Dead reckoning** | Estimating position by tracking movement from a known start point |
| **Delta** | River delta — fan-shaped sediment deposit where river meets body of water |
| **Depth map** | Image where each pixel value = distance to that surface point |
| **DriveAPI** | USD/PhysX API that turns a passive joint into a controllable motor |
| **DTM** | Digital Terrain Model — elevation data for a geographic area |
| **D* Lite** | Incremental path planning algorithm used by Mars rovers |
| **FOV** | Field of View — how wide an angle the camera captures |
| **FSM** | Finite State Machine — a system that can be in one of several states |
| **GeoTIFF** | Image format that embeds geographic coordinate information |
| **GPU** | Graphics Processing Unit — runs Isaac Sim, renders images, runs AI models |
| **HazCam** | Hazard Avoidance Camera — wide-angle cameras for obstacle detection |
| **HiRISE** | High Resolution Imaging Science Experiment — camera on Mars Reconnaissance Orbiter |
| **ICRA** | International Conference on Robotics and Automation — top robotics venue |
| **Isaac Sim** | NVIDIA's GPU-accelerated robotics simulator |
| **Jezero Crater** | 49km wide impact crater on Mars, Perseverance's landing site |
| **Joint** | Physics connection between two rigid bodies allowing constrained movement |
| **MastCam-Z** | Perseverance's main science camera on the mast, with zoom capability |
| **MOLA** | Mars Orbiter Laser Altimeter — global Mars elevation map |
| **NavCam** | Navigation Camera — used for driving and basic science imaging |
| **Node** | A ROS2 program that participates in the message bus |
| **Odometry** | Estimating robot position/velocity by tracking wheel/sensor motion |
| **OmniGraph** | Isaac Sim's visual programming system for wiring sensor outputs to ROS2 |
| **Prim** | A single element in a USD scene (mesh, camera, light, transform, etc.) |
| **Publisher** | A ROS2 node that sends messages on a topic |
| **QoS** | Quality of Service — how reliable/lossy a ROS2 topic connection is |
| **Regolith** | Loose unconsolidated surface material (dust, sand, broken rock) on Mars |
| **RELIABLE** | ROS2 QoS policy: every message guaranteed to arrive |
| **RevoluteJoint** | A physics joint that allows rotation around exactly one axis |
| **Rigid body** | A physics object that doesn't deform — all Mars rover parts |
| **Rocker-bogie** | Mars rover suspension design: pivoting beams keep all 6 wheels on ground |
| **ROS2** | Robot Operating System 2 — messaging framework for robot software |
| **RTG** | Radioisotope Thermoelectric Generator — nuclear power source on Perseverance |
| **Sol** | One Martian day = 24 hours 37 minutes |
| **Stage** | The complete USD scene — all objects, materials, physics |
| **Subscriber** | A ROS2 node that reads messages from a topic |
| **TF2** | ROS2 transform library — tracks coordinate frames of all robot parts |
| **Topic** | A named channel in ROS2 for sending/receiving typed messages |
| **Traversability** | How safe/driveable a piece of terrain is (0=safe, 1=impassable) |
| **USD** | Universal Scene Description — Pixar's 3D scene format, used by Isaac Sim |
| **VLM** | Vision-Language Model — AI that understands both images and text questions |
| **VRAM** | Video RAM — GPU memory. RTX 3090 has 24 GB. |
| **Xform** | USD transform prim — holds position, rotation, scale for child prims |

---

*Handbook maintained alongside the codebase. Update this file whenever a new concept is introduced to the system.*

*Last updated: 2026-05-26 — Stages 1 & 2 complete (terrain + rover)*
