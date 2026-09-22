# Multi-product kitting

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

## Setup

TODO — document once the pipeline runs on this IPC.
EOF