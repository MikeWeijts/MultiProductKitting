# -----------------------
# Multi-product kitting
# -----------------------

Kitting playsets onto pallets with a Doosan H2017 at the Brainport Industries
Campus. Because new playsets are released regularly, a conventional setup has
to be reprogrammed for every new product. The aim here is a system that
generalises across product variants: a Mech-Eye 3D camera provides visual
input, an Octo VLA policy produces end-effector actions, and ROS2 drives the
robot in both Isaac Sim and the physical cell.

Forked from [DuncanKikkert1/ootf_ros2](https://github.com/DuncanKikkert1/ootf_ros2),
which established the Isaac Sim → TFDS → Octo fine-tune → ROS2 pipeline for
single pick-and-place. This repository extends that toward kitting a complete
product set.

## Status

Work in progress. See [docs/upstream-README.md](docs/upstream-README.md) for
the inherited setup and pipeline documentation.

## Running

Everything starts through `run.sh`. Run `./run.sh help` for all modes and flags.

```bash
# Start only the sim and check that the scene loads
./run.sh debug sim

# Collect demos, convert them to TFDS and finetune Octo
./run.sh pipeline --output-dir data/exp_01 --preset phased-cube

# Run the trained policy in the sim
./run.sh sim --instruction "pick up the cube and place it on the conveyor"

# Measure the closed-loop success rate over 20 attempts
./run.sh debug success --attempts 20

# Run the policy on the real robot
./run.sh real --instruction "pick up the cube and place it on the conveyor"
```

`sim` and `real` use the newest finetuned checkpoint automatically, or the
pretrained Octo model if there is none. Logs are written to `debug/logs/run/`.

For the full flag reference, the DAgger workflow, finetuning settings and the
TCP message formats, see [docs/upstream-README.md](docs/upstream-README.md).
It's Installation section is outdated for this repo, so use the Setup section
below instead.

## Setup

Tested on the IPC with:

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 |
| ROS2 | Jazzy |
| NVIDIA driver and CUDA | 580, CUDA 12 |
| Python | 3.11.14, virtualenv `3.11.14/envs/python3.11` |
| Isaac Sim | 5.1.0 (pip) |

Isaac Sim 5.1 only supports Python 3.11. Other versions fail with errors that
don't point at the Python version, so stick to the one below.

### 1. ROS2 Jazzy

Follow the [official install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
The launch scripts find it in `/opt/ros/` automatically.

### 2. Python environment

Everything (Isaac Sim, Octo, JAX, TensorFlow) is installed into one pyenv
virtualenv, created from Python 3.11.14 with the name `python3.11`:

```bash
pyenv install 3.11.14
pyenv virtualenv 3.11.14 python3.11
```

pyenv refers to this environment as `3.11.14/envs/python3.11`, which is what
[.python-version](.python-version) contains. Inside the repo folder, `python`
and `pip` therefore use this environment automatically. Check with:

```bash
python -c "import isaacsim; print('ok')"   # after step 3
```

Use exactly these names. With a different version or name, `.python-version`
won't match and you get a bare Python without the packages.

The launch scripts don't depend on `.python-version`. They search
`~/.pyenv/versions` for a Python that can import `isaacsim`, or `jax` and
`octo`.

### 3. Isaac Sim 5.1

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 4. Isaac Sim ROS2 workspace

```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git ~/IsaacSim-ros_workspaces
```

Build it for Jazzy following the instructions in that repo. The launch scripts
look for `~/IsaacSim-ros_workspaces/build_ws/jazzy`. For another location, set
`ISAAC_WS=/path/to/workspace`.

### 5. Octo and JAX

```bash
git clone https://github.com/octo-models/octo.git ~/Documents/octo
cd ~/Documents/octo && git checkout 241fb35
pip install -e ~/Documents/octo
pip install "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install matplotlib   # policy diagnosis plots
```

Upstream Octo doesn't import with this JAX version. Make two small edits in
`~/Documents/octo`:

1. In `octo/utils/typing.py`, replace `PRNGKey = jax.random.KeyArray` with:
   ```python
   try:
       PRNGKey = jax.random.KeyArray
   except AttributeError:
       PRNGKey = jax.Array  # fallback for older/different JAX
   ```
2. In `octo/data/utils/data_utils.py`, add `from __future__ import annotations`
   as the first line, and change `dl.DLataset` to `dl.Dataset` in
   `get_dataset_statistics`.

Known good versions of the key packages, taken from the working IPC
environment:

| Package | Version |
|---|---|
| `jax`, `jaxlib` | 0.4.20, 0.4.20+cuda12.cudnn89 |
| `flax`, `optax`, `chex`, `distrax` | 0.7.5, 0.1.5, 0.1.85, 0.1.5 |
| `orbax-checkpoint` | 0.4.8 |
| `tensorflow`, `tensorflow-datasets`, `tensorflow-probability` | 2.15.0, 4.9.9, 0.23.0 |
| `numpy`, `scipy`, `ml-dtypes` | 1.26.0, 1.12.0, 0.2.0 |

### 6. Real robot only

```bash
pip install MechEyeAPI   # Mech-Eye camera SDK, 2.5.4 on the IPC
```

For the Doosan ROS2 package, follow [doosan-robot2](https://github.com/doosan-robotics/doosan-robot2).

### 7. Clone and check

```bash
git clone https://github.com/MikeWeijts/MultiProductKitting.git
cd MultiProductKitting
python -c "import isaacsim, jax, octo; print('ok')"
./run.sh debug sim
```

`./run.sh debug sim` should log `[SCENE] Built sim2_sim.usd`, `[ROBOT] 6 DOF`
and `[CAMERA] Ready`. Run logs are written to `debug/logs/run/`.

### Scene assets

Everything the simulation loads lives in [scenes/](scenes/) and is tracked in
git: the scene `scenes/usd/sim2.usd`, the floor and its textures, the SMC
gripper, the conveyor, the Mech-Eye camera mesh and the H2017 robot in
`scenes/h2017/urdf/h2017.blue/`. CAD source files (`.STEP`, `.stp`, `.usdz`
and the conveyor `.obj`) are excluded in `.gitignore` because the sim never
loads them.

`sim2.usd` itself loads the robot from an absolute path on the original
machine. [src/isaac/scene_prep.py](src/isaac/scene_prep.py) builds a cleaned
copy, `scenes/usd/sim2_sim.usd`, that loads the robot from the repo instead.
That copy is generated and not tracked. Delete it to force a rebuild after
changing `scene_prep.py`.

The robot uses "acceleration" joint drives, which is the version the existing
training data was collected with. Other imports of the H2017, for example in
ootf_ros2, use "force" drives. Don't swap in a robot USD from elsewhere
without collecting new data.