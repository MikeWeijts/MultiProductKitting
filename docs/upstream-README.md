# ootf_ros2

A ROS2 pipeline for training and deploying a Doosan H2017 robot arm using the
[Octo](https://github.com/octo-models/octo) Vision-Language-Action (VLA) model.
Synthetic demonstrations are collected in Isaac Sim, used to finetune Octo, and
the resulting policy drives the robot in simulation or on the real hardware.

---

## Architecture

### Data collection & training

```
Isaac Sim (headless)
  └─ collect.py          Scripted episodes with Replicator domain randomisation
       │  two task modes (see Task modes below):
       │    PickAndPlaceTask  — random pick + random place anywhere on conveyor
       │    SortingTask       — random pick + fixed per-object platform on belt
       │  inactive objects parked underground; SMC two-cup surface gripper
       │  EEF approach: straight-down for cube/cylinder, face-normal for pyramid
       │  saves .npz episodes  (images, actions, instruction, rng_state)
       ▼
  tfds_builder.py        Converts .npz → TFDS dataset  (ootf_synthetic)
       │
       ▼
  Octo finetune          full mode (default), wrist camera, language-conditioned
       │  saves checkpoint
       ▼
  data/<exp>/checkpoint/octo_finetune/experiment_<timestamp>/
```

### Task modes

| Mode | Class | Place behaviour |
|---|---|---|
| `random` | `PickAndPlaceTask` | Place position sampled uniformly from a configurable XY region |
| `sorting` | `SortingTask` | Each object type has a fixed target platform on the belt (cube x=−0.5, cylinder x=0, pyramid x=0.5; all at y=−1.28, z=0.855 m) |

### Sim inference

```
sim_node.py             Isaac Sim scene + ROS2 node
  publishes  /mecheye/color   (camera frames)
  subscribes /eef_delta       (EEF delta actions)
       ↕  ROS2
run_octo_live.py        Octo inference loop
  subscribes /mecheye/color
  publishes  /eef_delta
```

### Real robot inference

```
MechEye camera
  └─ run_octo_live.py   Octo inference → TCP EEF deltas (port 9005)
       │  TCP
       ▼
  tcp_ros_bridge.py     TCP receiver → publishes /joint_command
       │  ROS2
       ▼
  joint_service_client.py   rad→deg, calls /dsr01/motion/move_joint
       │  ROS2 service
       ▼
  Doosan H2017
```

---

## Repository structure

```
ootf_ros2/
├── run.sh                          Single entry point for all modes
├── launch/
│   ├── pipeline.sh                 Collect → convert → finetune
│   ├── sim_inference.sh            Isaac Sim + Octo VLA
│   ├── real_inference.sh           Real robot + Octo VLA
│   └── debug.sh                    Individual component launcher + policy diagnosis
├── src/
│   ├── isaac/
│   │   ├── collect.py              Headless episode collection in Isaac Sim
│   │   ├── sim_node.py             Isaac Sim ROS2 node (inference-time)
│   │   ├── task.py                 PickAndPlaceTask + SortingTask; LULA IK waypoint sampling
│   │   └── gripper.py              SMC two-cup surface gripper controller (Isaac SurfaceGripper)
│   ├── vla/
│   │   ├── octo_policy.py          Octo model wrapper (inference)
│   │   ├── run_octo_live.py        Live Octo inference loop
│   │   ├── mech_eye_camera.py      MechEye SDK and ROS2 camera interfaces
│   │   ├── tcp_ros_bridge.py       TCP receiver → /joint_command publisher
│   │   ├── tcp_sender.py           EEF delta sender (TCP or ROS2)
│   │   ├── joint_service_client.py /joint_command → Doosan MoveJoint service
│   │   └── axis_check.py           Utility to verify axis directions
│   └── training/
│       ├── pipeline.py             Orchestrates convert + finetune steps
│       ├── finetune_config.py      Octo finetune configuration
│       └── tfds_builder.py         .npz → TFDS dataset builder
├── debug/
│   ├── debug_policy.py             Policy diagnosis script
│   ├── success_rate.py             Closed-loop success-rate harness (+ --record-dagger)
│   ├── dagger_relabel.py           DAgger oracle relabeler (corrective episodes)
│   ├── test_gripper.py             Surface gripper open/close test (sim running)
│   ├── visualize_episode.py        Plot and inspect collected .npz episodes
│   ├── policy_diagnosis/           PNG + TXT diagnosis reports (git-ignored)
│   └── logs/                       Runtime session logs (git-ignored)
├── scenes/
│   ├── usd/doosan_BIC.usd          Isaac Sim scene
│   └── h2017/urdf/
│       ├── h2017.urdf              Doosan H2017 URDF
│       └── h2017_lula.yaml         LULA kinematics config (auto-generated)
└── data/                           Episodes, TFDS datasets, checkpoints (git-ignored)
    └── <exp>/
        ├── raw/                    Collected .npz episode files
        ├── tfds/                   Converted TFDS dataset
        └── checkpoint/             Octo finetune checkpoints
```

---

## Installation

### Requirements
- Ubuntu 22.04 or 24.04
- ROS2 (Humble / Jazzy)
- Python 3.11 (Isaac Sim requirement — use pyenv)
- NVIDIA GPU with CUDA 12

### 1. ROS2
Follow the official guide for your distro:
[Humble (22.04)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) |
[Jazzy (24.04)](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

### 2. Python 3.11 via pyenv
```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 python3.11
pyenv global python3.11
```

### 3. Isaac Sim
```bash
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
            isaacsim-extscache-kit isaacsim-extscache-kit-sdk \
            --extra-index-url https://pypi.nvidia.com
```

### 4. Isaac Sim ROS2 workspace
```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git ~/IsaacSim-ros_workspaces
# Follow the build instructions in that repo for your ROS2 distro
```

### 5. Octo + JAX
```bash
git clone https://github.com/octo-models/octo.git ~/Documents/octo
pip install -e ~/Documents/octo
pip install "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install matplotlib  # for policy diagnosis plots
```

### 6. MechEye SDK (real robot only)
```bash
pip install MechEyeAPI
```

### 7. Doosan ROS2 package (real robot only)
Follow: https://github.com/doosan-robotics/doosan-robot2

### 8. Clone this repo
```bash
git clone https://github.com/DuncanKikkert1/ootf_ros2.git
cd ootf_ros2
chmod +x run.sh
```

---

## Usage

All modes are accessed through `run.sh`. Run `./run.sh help` for the condensed
quick reference, or `--help` on any underlying script for its full flag list.

```bash
./run.sh <mode> [args]
```

### Modes & flags

**Pipeline** — `./run.sh pipeline --output-dir <dir> [phase] [flags]`

| Phase (default = full) | Effect |
|---|---|
| *(none)* | collect + convert + finetune |
| `--collect-only` / `--convert-only` / `--finetune-only` | single phase |
| `--train-head-only` / `--collect-and-train-head` | linear head instead of finetune |

| Group | Flags (defaults in parens) |
|---|---|
| Shorthand | `--preset phased-cube` = `--phased --forced-obj-sequence cube` |
| Collect | `--n-episodes` (200) · `--seed` · `--phased` · `--overfit` · `--gui` · `--task-type random\|sorting` · `--forced-obj-sequence OBJ…` · `--domain-rand` · `--fixed-yaw RAD` · `--instruction` |
| Train | `--finetune-mode full\|head_only\|head_mlp_only` (**full**) · `--task-modality text_conditioned\|image_conditioned\|multimodal` (text) · `--n-finetune-steps` (50000) · `--batch-size` (32) · `--overfit` |
| Head | `--head-model-path` · `--head-batch-size` (32) · `--lam` (1e-4) |
| DAgger | `--dagger-rollouts <dir>` (relabel + aggregate instead of collect) · `--expert-raw <dir>` (default `data/exp_truecolor2/raw`) · `--dagger-phase` (`pick`) — see [DAgger workflow](#dagger--correcting-closed-loop-drift) |

**Inference** — `./run.sh sim\|real [flags]`

| Flag | Purpose |
|---|---|
| `--instruction "…"` / `--goal-image PNG` | task spec (mutually exclusive) |
| `--head-path <dir>/linear_head.npz` | use linear head instead of diffusion head (recommended) |
| `--phased` · `--phase-object cube` | phase-switching rollout |
| `--zero-rotation` | zero the rotation dims (yaw-symmetric objects) |
| `--temporal-ensemble` | action smoothing — **off** by default; `sim` runs raw, `real` enables it |
| `--grip-threshold` (0.9) · `--grip-open-threshold` (0.5) · `--grip-open-steps` (3) · `--grip-hold-steps` (20) | gripper schmitt-trigger |

**Debug** — `./run.sh debug <sub>`

| Sub | Purpose |
|---|---|
| `sim` / `bridge` / `joint` / `gripper-test` | run one component in isolation |
| `collect [--seq seq1-seq4] [--seed N]` | dry-run collector (writes nothing) |
| `policy [--n-episodes N] [--pretrained]` | open-loop policy vs ground truth (image → action) |
| `replay [--npz <episode.npz>]` | GT action replay through the sim (action → motion) |
| `success [--attempts N] [--phased] …` | closed-loop success rate over N resets |

**Env toggles:** `OOTF_PROPRIO=0\|1` (proprio tokenizer, default on) · `OOTF_SORT_PLATFORMS=1` (spawn sort platforms) · `OOTF_RECORD_DIR=<dir>` / `OOTF_RECORD_EVERY=N` (frame capture).

### Full pipeline — collect + convert + finetune
```bash
./run.sh pipeline --output-dir data/exp_01 --preset phased-cube   # canonical run
./run.sh pipeline --output-dir data/exp_01 --n-episodes 200       # custom
```

Run individual phases:
```bash
./run.sh pipeline --output-dir data/exp_01 --collect-only    # collect only
./run.sh pipeline --output-dir data/exp_01 --convert-only    # .npz → TFDS only
./run.sh pipeline --output-dir data/exp_01 --finetune-only   # finetune only
```

### Overfit sanity check
To prove the pipeline works end to end, train on episodes that are *fully*
identical and verify the model replays them. Both flags are required —
`--overfit` on collection pins every per-episode sampling source (object type,
instruction, yaw, pick **and place** position, domain rand); `--overfit` on
finetune disables augmentation/weight decay and keeps every dwell frame:

```bash
./run.sh pipeline --output-dir data/exp_overfit --n-episodes 20 --overfit \
    --finetune-mode full --n-finetune-steps 20000
./run.sh debug policy        # open-loop: correlations ≈ 1 expected on train data
./run.sh debug replay        # controller-only: GT actions must track within ~2 cm
./run.sh sim --instruction "pick up the cube and place it on the conveyor"
    # closed-loop replay of the memorised trajectory (raw chunk[0] output is the
    # default; add --temporal-ensemble only if you want action smoothing)
```

When running the phases separately, all three are needed and **each** needs
`--overfit` (collect: deterministic episodes; convert: keep dwell frames;
finetune: no augmentation, zero weight decay):

```bash
./run.sh pipeline --output-dir data/exp_overfit --collect-only  --n-episodes 20 --overfit
./run.sh pipeline --output-dir data/exp_overfit --convert-only  --overfit
./run.sh pipeline --output-dir data/exp_overfit --finetune-only --overfit \
    --finetune-mode full --n-finetune-steps 20000
```

Fixing only `--pick-x/--pick-y/--fixed-yaw/--no-domain-rand` is **not**
sufficient: object type, language instruction, and place position are still
sampled per episode, giving the model several different behaviours to
memorise under one task.

### DAgger — correcting closed-loop drift

The collected expert demos are optimal-state trajectories, so a finetuned policy
drifts in closed loop (covariate shift) and misses the grasp. DAgger closes the
gap: record the policy's *own* rollout states, relabel them with the sim oracle's
recovery action, and finetune on the expert ∪ corrective set. Three steps:

```bash
# 1. Record on-policy rollout traces while measuring the current model
#    (needs a running sim: ./run.sh debug sim)
./run.sh debug success --model-path <ckpt> --phased --phase-object cube \
    --attempts 50 --record-dagger data/exp_dagger/rollouts

# 2. Relabel + aggregate + convert + finetune (no Isaac collect needed).
#    Pass the SAME base and proprio setting as the model you are improving —
#    the DAgger finetune inherits NEITHER automatically.
OOTF_PROPRIO=1 ./run.sh pipeline --output-dir data/exp_dagger \
    --dagger-rollouts data/exp_dagger/rollouts \
    --pretrained-path <base-checkpoint> \
    --finetune-mode full --task-modality multimodal --n-finetune-steps 50000

# 3. Re-evaluate the DAgger model against the same harness
./run.sh debug success \
    --model-path data/exp_dagger/checkpoint/octo_finetune/experiment_<ts> \
    --phased --phase-object cube --attempts 50
```

The relabeler (`debug/dagger_relabel.py`) only writes a gripper-close label when
the rollout actually reached the grasp point, so corrective episodes teach the
**approach**; the grasp demonstrations themselves come from the expert set. The
expert set merged with the corrective episodes defaults to
`--expert-raw data/exp_truecolor2/raw`; the relabeled phase defaults to
`--dagger-phase pick`.

> **Gotcha — proprio.** `--pretrained-path` and `OOTF_PROPRIO` are independent of
> the expert set. Finetuning from an image-only base without `OOTF_PROPRIO=1`
> silently produces an image-only model that flails and rams in closed loop.
> Before trusting a DAgger run, confirm the new checkpoint's `config.json`
> contains a `proprio` tokenizer (`grep -c proprio <ckpt>/config.json`) and that
> inference does **not** warn `'observations' contains extra items … {'proprio'}`.

### Sim inference — Isaac Sim + Octo VLA
```bash
./run.sh sim --instruction "pick up the object"
```

### Real robot inference — MechEye + Octo + Doosan
```bash
./run.sh real --instruction "pick up the object"
```

### Debug & diagnostics
```bash
./run.sh debug policy              # diagnose finetuned policy vs ground truth
./run.sh debug policy --n-episodes 20
./run.sh debug policy --pretrained # compare against base pretrained model
./run.sh debug policy --step 10000 # evaluate a specific checkpoint step

./run.sh debug replay              # replay GT actions through the sim (action → motion)
./run.sh debug replay --npz data/exp_01/raw/episode_000000.npz

./run.sh debug sim                 # start Isaac Sim node in isolation
./run.sh debug bridge              # start TCP→ROS2 bridge in isolation
./run.sh debug joint               # start Doosan joint client in isolation
./run.sh debug gripper-test        # open/close gripper while sim is running

# Dry-run episode collector — visualises trajectories without writing .npz files
./run.sh debug collect                                    # 3 random episodes
./run.sh debug collect --seq seq1                         # 1 cube episode
./run.sh debug collect --seq seq2                         # 1 cylinder episode
./run.sh debug collect --seq seq3                         # 1 pyramid episode
./run.sh debug collect --seq seq4                         # cube → cylinder → pyramid
./run.sh debug collect --task-type sorting --seq seq4     # sorting task, all three
./run.sh debug collect --seq seq4 --seed 7                # reproducible seed
./run.sh debug collect --from-episode data/exp/raw/episode_000001.npz  # replay
```

Runtime logs for `sim`, `bridge`, and `joint` are saved to `debug/logs/` with a
timestamp. Policy diagnosis reports (PNG plot + stats table) are saved to
`debug/policy_diagnosis/` named after the checkpoint and data used.

---

## Finetuning

### Key configuration — `src/training/finetune_config.py`

| Parameter | Default | Notes |
|---|---|---|
| `max_steps` | 50,000 | ~45 passes through 200 episodes |
| `window_size` | 2 | Consecutive frames passed to model — gives temporal context |
| `finetuning_mode` | `full` (set by the pipeline CLI) | What weights train — see Finetuning modes below |
| `batch_size` | 32 | Keep at 32; larger batches risk overfitting on small datasets |
| `peak_value` (LR) | 3e-4 | Cosine schedule with 2000-step warmup |

### Finetuning modes

| Mode | What trains | When to use |
|---|---|---|
| `full` | Entire model | **Default — what the real runs use** |
| `head_only` | Full action head (attention + MLP) | Lighter; risks overwriting pretrained attention |
| `head_mlp_only` | Final MLP layers only | Cheapest probe; freezes transformer + attention |

### Recommended dataset size
- Minimum: 200 episodes (below this, 95/5 train/val split may fail)
- Recommended: 400+ episodes for better workspace coverage

---

## TCP message formats

### Joint commands — `tcp_ros_bridge.py` (port 9002)
```
key_cmd;status;extra_num;gripper;x;y;z;aw;ap;ar
```
Fields 1–3 are integers; fields 4–10 are floats (joint positions in radians).

### EEF delta actions — `tcp_sender.py` (port 9005)
```
dx;dy;dz;drx;dry;drz;gripper\n
```

| Field | Unit | Description |
|---|---|---|
| `dx` `dy` `dz` | metres | End-effector position delta |
| `drx` `dry` `drz` | radians | End-effector orientation delta (XYZ Euler) |
| `gripper` | 0.0–1.0 | 0 = open, 1 = closed |

---

## Verifying a running pipeline

```bash
# Check camera frames are publishing (sim mode)
ros2 topic echo /mecheye/color --no-arr

# Check EEF delta actions are publishing
ros2 topic echo /eef_delta

# Check joint commands are reaching the bridge
ros2 topic echo /joint_command

# Check joint states from sim
ros2 topic echo /joint_states
```
