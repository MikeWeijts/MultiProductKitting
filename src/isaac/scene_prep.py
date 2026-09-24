# scene-prep.py - Loads the scene. Previously this was one seperately in sim_node.py and collect.py which created duplicate code and if you want to change one of them you have to change both or your training or video generation uses different data.
# To prevent this I made a seperate file scene_prep to load in the scenes.

from pathlib import Path
from pxr import Usd, Sdf

def build_clean_scene(raw_scene):
    _RAW_SCENE = Path(raw_scene)
    _CLEAN_SCENE = _RAW_SCENE.parent / (_RAW_SCENE.stem + "_sim.usd")

    if (not _CLEAN_SCENE.exists() or
            _RAW_SCENE.stat().st_mtime > _CLEAN_SCENE.stat().st_mtime):
        _stage = Usd.Stage.Open(str(_RAW_SCENE))
        _paths = [p.GetPath() for p in _stage.Traverse()
                if p.GetTypeName() in ("OmniGraph", "ComputeGraph")]
        for _path in _paths:
            _stage.RemovePrim(_path)

        # sim2.usd loads the robot from an absolute path on one PC (~/Documents/temp/...). Pointed it at the copy in the repo instead, so the scene works after a fresh clone. The path is relative to the sim.usd file (scenes/usd/).
        _robot = _stage.GetRootLayer().GetPrimAtPath("/World/h2017")
        _robot.referenceList.prependedItems = [
            Sdf.Reference("../h2017/urdf/h2017.blue/h2017.blue.usd")
        ]

        _stage.GetRootLayer().Export(str(_CLEAN_SCENE))
        print(f"[SCENE] Built {_CLEAN_SCENE.name}: removed {len(_paths)} OmniGraph prim(s)")
    else:
        print(f"[SCENE] Using cached {_CLEAN_SCENE.name}")

    return _CLEAN_SCENE