# camera_checks.py — guard against Isaac substituting a default camera prim.

def require_camera_prim(prim_path: str) -> None:
    """Raise if prim_path doesn't already exist

    Isaac's Camera() defines a NEW default Camera prim when prim_path doesn't
    resolve, logging only at carb.log_info — below the default threshold. The
    substitute has the wrong pose and intrinsics, and the policy reads from it
    without complaint. 
    """
    from isaacsim.core.utils.prims import is_prim_path_valid

    if not is_prim_path_valid(prim_path):
        raise RuntimeError(
            f"Camera prim missing: {prim_path}\n"
            "Isaac would silently substitute a default camera here."
            "means a payload failed to resolve — check for 'Could not open asset' "
            "warnings earlier in this log."
        )