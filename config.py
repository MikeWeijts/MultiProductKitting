# config.py - Shared settings for sim_node.py, collect.py and scene_prep.py.
# Define a value here once instead of in every script, so collection and
# evaluation can't drift apart.

from pathlib import Path

# Repo root, found from this file's location
REPO_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------
# Robot H2017
# -----------------------------------------------
ROBOT_PRIM  = "/World/h2017"
EE_FRAME    = "link_6"
H2017_DIR   = REPO_ROOT / "scenes" / "h2017"
URDF_PATH   = H2017_DIR / "urdf" / "h2017.urdf"
LULA_DESC   = H2017_DIR / "urdf" / "h2017_lula.yaml"

# -----------------------------------------------
# Scene Files
# -----------------------------------------------
SCENE_USD = REPO_ROOT / "scenes" / "usd" / "sim2.usd"
ROBOT_USD = H2017_DIR / "urdf" / "h2017.blue" / "h2017.blue.usd"

# -----------------------------------------------
# Camera MechEye
# -----------------------------------------------
# Used different type of definition because it doesn't start with a Path like the others
WRIST_CAM_PRIM = f"{ROBOT_PRIM}/{EE_FRAME}/MechEye/MechEye/Camera"