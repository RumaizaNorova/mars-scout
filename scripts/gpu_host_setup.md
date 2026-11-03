# Running ARES on a GPU host

## What to rent

- Linux VM or bare metal. Docker has to work, and many "container" rentals block it. `scripts/check_host.sh` tells you within a minute.
- NVIDIA RTX GPU with 16+ GB VRAM (24 GB is better), for example an RTX 3090 or 4090.
- 32 GB RAM and 100+ GB of SSD. The Isaac image is large and the terrain is 1.8 GB.
- Docker and the NVIDIA Container Toolkit installed, and a driver that meets NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

Rent for an hour first. Stop the host when you finish, because billing usually continues while it's on.

## Steps

```bash
git clone https://github.com/RumaizaNorova/mars-scout.git
cd mars-scout
cp .env.example .env        # read the Isaac licence, set ACCEPT_EULA=Y, add your keys
bash scripts/download_hirise.sh
bash scripts/smoke_test.sh
```

The smoke test passes only if all of these hold:

- Gravity is set to 3.72 m/s², and the rover really falls at that rate.
- The rover lands and comes to rest.
- The camera frame isn't black or blank.
- The rover drives when commanded inside Isaac Sim.
- Camera, depth and odometry reach the ROS container, and a ROS command moves the rover.

Then choose a Gemini model and run:

```bash
docker compose --profile sim run --rm ros python3 scripts/list_gemini_models.py
# put one model ID in .env as GEMINI_MODEL
bash scripts/run_demo.sh "dark olivine boulder"
```

## When something fails

| Symptom | Where to look |
|---|---|
| `check_host` says Docker is not usable | This is a container rental. Rent a VM or bare-metal host |
| `check_host` says containers cannot see the GPU | Install the NVIDIA Container Toolkit |
| Scene exits with an error | The error names the failed step. Scene problems now stop the run on purpose |
| Nothing arrives in `ros-smoke` | Check that `ROS_DOMAIN_ID` matches and both containers use host networking |
| `GEMINI_MODEL` error | Set it from the list above. Google retires model IDs often |

## Changing the Isaac Sim version

Update the tag and digest in `docker/isaac.Dockerfile`, then run the smoke test. Isaac 6.x uses Python 3.12, so regenerate `requirements/isaac.txt` as well.
