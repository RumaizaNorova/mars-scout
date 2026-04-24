## GPU host setup (AWS / Ubuntu) — draft checklist

This is a lightweight checklist we will convert into scripts once the exact AWS instance + Ubuntu version is chosen.

### Goal

On the GPU host we need:

- Ubuntu (x86_64)
- NVIDIA driver working (`nvidia-smi`)
- ROS 2 installed (Humble or Jazzy; pinned)
- Isaac Sim installed (pinned version)
- Repo cloned + colcon build

### AWS notes (practical)

- Prefer an RTX-backed instance (e.g. G5 family) and enough disk for Isaac + caches + bags.
- Use an AMI that makes driver install simple (NVIDIA GPU AMI), or install NVIDIA drivers explicitly.

### Verification commands (manual)

- `nvidia-smi` (GPU visible)
- `python3 --version`
- `ros2 doctor` (ROS healthy)
- Start Isaac, load the stage, confirm camera sensors can publish to ROS.

