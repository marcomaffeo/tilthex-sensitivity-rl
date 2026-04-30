"""Animate saved rollout trajectories using the MuJoCo renderer.

For each trajectory group (identified by trajectory number) in the input
directory all samples are rendered simultaneously in the same scene:
  - Nominal trajectory (traj_NNNNN.msgpack or traj_NNNNN_sample_00.msgpack)
    → opaque drone body
  - Perturbed trajectories (traj_NNNNN_sample_NN.msgpack, NN > 0)
    → semi-transparent drone body

Output: one MP4 per trajectory group, written next to the input files.

Usage
-----
# Single trajectory group (all files matching traj_00001*.msgpack)
python animate_rollout.py runs/.../rollouts/ --traj 1

# All trajectory groups in a directory
python animate_rollout.py runs/.../rollouts/

# Override output location
python animate_rollout.py runs/.../rollouts/ --output /tmp/videos/

Options
-------
--camera   track_side | top_down  (default: track_side)
--fps      render fps (default: 30)
--speed    playback speed multiplier (default: 1.0; >1 = faster)
--width    frame width  (default: 1280)
--height   frame height (default: 720)
--alpha    transparency for perturbed drones (default: 0.1)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
import copy
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Select an offscreen OpenGL backend if none is set.  EGL works on headless
# GPU servers; OSMESA is the CPU software fallback.  This must be done before
# importing mujoco.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from flax import serialization

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

# ---------------------------------------------------------------------------
# Mesh path
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "envs" / "assets"
_SCENE_XML  = _ASSETS_DIR / "scene.xml"
_DRONE_XML  = _ASSETS_DIR / "tilthex_laas.xml"
_MESH_PATH  = _ASSETS_DIR / "merged" / "tilthex_laas__26d05b7734899ce35f05c512c442e698_visual.stl"


# ---------------------------------------------------------------------------
# Rollout loading
# ---------------------------------------------------------------------------

def load_rollout(path: Path) -> Dict[str, np.ndarray]:
    """Load a msgpack rollout file and return a dict of numpy arrays."""
    raw = serialization.msgpack_restore(path.read_bytes())

    def _to_numpy(v):
        if isinstance(v, dict):
            return {k2: np.array(v2, dtype=np.float64) for k2, v2 in v.items()}
        return np.array(v, dtype=np.float64)

    return {k: _to_numpy(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# File grouping
# ---------------------------------------------------------------------------

_TRAJ_RE = re.compile(r"^traj_(\d+)(?:_sample_(\d+))?\.msgpack$")


def group_rollout_files(
    rollout_dir: Path,
) -> Dict[int, List[Tuple[Path, bool]]]:
    """Return {traj_id: [(path, is_nominal), ...]}.

    is_nominal=True  → traj_NNNNN_sample_00.msgpack
    is_nominal=False → traj_NNNNN_sample_NN.msgpack with NN > 0

    If _sample_ files exist for a traj_id, the bare traj_NNNNN.msgpack is
    skipped (it would be a duplicate of sample_00).
    """
    groups: Dict[int, List[Tuple[Path, bool]]] = defaultdict(list)
    bare: Dict[int, Path] = {}  # traj_id → bare file (no _sample_ suffix)

    for f in sorted(rollout_dir.iterdir()):
        m = _TRAJ_RE.match(f.name)
        if m is None:
            continue
        traj_id = int(m.group(1))
        sample_id = int(m.group(2)) if m.group(2) is not None else None
        if sample_id is None:
            bare[traj_id] = f
        else:
            is_nominal = sample_id == 0
            groups[traj_id].append((f, is_nominal))

    # For traj_ids that have no _sample_ files, fall back to the bare file
    for traj_id, f in bare.items():
        if traj_id not in groups:
            groups[traj_id].append((f, True))

    return dict(groups)


# ---------------------------------------------------------------------------
# MuJoCo XML builder
# ---------------------------------------------------------------------------

_DRONE_RGBA_OPAQUE = "0.25 0.0 0.0 1.0"
_DRONE_RGBA_TRANSPARENT = "0.25 0.25 0.25 {alpha}"


_TRAIL_STEP = 4       # sample every N sim steps for trail connectors
_TRAIL_RADIUS = 0.008


def _camera_xyaxes(cam_pos: np.ndarray, cam_target: np.ndarray) -> str:
    """Compute MuJoCo xyaxes string for a camera at cam_pos looking at cam_target."""
    z = cam_pos - cam_target          # camera Z points out of screen (backward)
    z = z / np.linalg.norm(z)
    world_up = np.array([0.0, 0.0, 1.0])
    x = np.cross(world_up, z)         # right
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / np.linalg.norm(x)
    y = np.cross(z, x)                # up
    y = y / np.linalg.norm(y)
    return f"{x[0]:.4f} {x[1]:.4f} {x[2]:.4f} {y[0]:.4f} {y[1]:.4f} {y[2]:.4f}"


def build_mjcf(
    nominal_count: int,
    perturbed_count: int,
    has_obstacle: bool,
    has_target: bool,
    alpha: float,
    width: int = 1920,
    height: int = 1080,
    cam_pos: Optional[np.ndarray] = None,
    cam_target: Optional[np.ndarray] = None,
) -> str:
    """Build MJCF XML by modifying the base scene with runtime values."""
    if cam_pos is None:
        cam_pos = np.array([0.0, -5.0, 3.0])
    if cam_target is None:
        cam_target = np.array([0.0, 0.0, 1.0])

    xyaxes = _camera_xyaxes(cam_pos, cam_target)
    dist = float(np.linalg.norm(cam_pos - cam_target))
    cx, cy, cz = float(cam_target[0]), float(cam_target[1]), float(cam_target[2])

    td_height = float(min(max(dist * 8, 20.0), 150.0))
    td_half_fov_rad = np.arctan((dist * 0.6) / td_height)
    td_fovy = float(np.degrees(td_half_fov_rad) * 2.0)

    tree = ET.parse(_SCENE_XML)
    root = tree.getroot()

    # Remove include tag to prevent duplicate static drone
    include_tag = root.find("include")
    if include_tag is not None:
        root.remove(include_tag)

    # Parse drone xml and inject assets/defaults
    drone_tree = ET.parse(_DRONE_XML)
    drone_root = drone_tree.getroot()
    
    scene_asset = root.find("asset")
    for asset in drone_root.find("asset"):
        scene_asset.append(asset)
        
    drone_default = drone_root.find("default")
    if drone_default is not None:
        root.insert(1, drone_default)

    drone_template = drone_root.find(".//body[@name='tilthex_laas__26d05b7734899ce35f05c512c442e698']")

    stat = root.find("statistic")
    if stat is not None:
        stat.set("center", f"{cx:.2f} {cy:.2f} {cz:.2f}")
        stat.set("extent", f"{dist:.2f}")

    global_el = root.find("visual/global")
    if global_el is not None:
        global_el.set("offwidth", str(width))
        global_el.set("offheight", str(height))

    worldbody = root.find("worldbody")

    cam_side = ET.SubElement(worldbody, "camera", {"name": "track_side"})
    cam_side.set("pos", f"{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}")
    cam_side.set("xyaxes", xyaxes)

    cam_td = ET.SubElement(worldbody, "camera", {"name": "top_down"})
    cam_td.set("pos", f"{cx:.3f} {cy:.3f} {cz + td_height:.1f}")
    cam_td.set("fovy", f"{td_fovy:.3f}")

    if has_obstacle:
        obs_body = ET.SubElement(worldbody, "body", {"name": "obstacle", "mocap": "true"})
        ET.SubElement(obs_body, "geom", {"name": "obstacle_geom", "type": "box", "size": "0.5 0.5 1.5", "rgba": "0.5 0.5 0.5 1"})

    if has_target:
        tgt_body = ET.SubElement(worldbody, "body", {"name": "target", "mocap": "true"})
        ET.SubElement(tgt_body, "geom", {"type": "sphere", "size": "0.1", "rgba": "1 0 0 1"})

    perturbed_rgba = _DRONE_RGBA_TRANSPARENT.format(alpha=alpha)
    
    for i in range(nominal_count):
        body = copy.deepcopy(drone_template)
        body.set("name", f"drone_nominal_{i}")
        joint = body.find("freejoint")
        joint.set("name", f"drone_nominal_{i}_joint")
        for site in body.iter("site"):
            site.set("name", f"{site.get('name')}_nominal_{i}")
        for geom in body.iter("geom"):
            geom.set("rgba", _DRONE_RGBA_OPAQUE)
        worldbody.append(body)

    for i in range(perturbed_count):
        body = copy.deepcopy(drone_template)
        body.set("name", f"drone_perturbed_{i}")
        joint = body.find("freejoint")
        joint.set("name", f"drone_perturbed_{i}_joint")
        for site in body.iter("site"):
            site.set("name", f"{site.get('name')}_perturbed_{i}")
        for geom in body.iter("geom"):
            geom.set("rgba", perturbed_rgba)
        worldbody.append(body)

    return ET.tostring(root, encoding="unicode")

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _set_drone_qpos(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    joint_name: str,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    """Set the freejoint qpos for a single drone body."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    addr = model.jnt_qposadr[joint_id]
    data.qpos[addr : addr + 3] = pos
    # MuJoCo freejoint quaternion order is [w, x, y, z] — same as our saved attitude
    data.qpos[addr + 3 : addr + 7] = quat_wxyz


def render_trajectory_group(
    group: List[Tuple[Path, bool]],
    camera: str,
    fps: int,
    speed: float,
    width: int,
    height: int,
    alpha: float,
    env_dt: float = 0.02,
) -> np.ndarray:
    """Render all trajectories in a group; return (T, H, W, 3) uint8 array."""
    # Load all rollouts
    rollouts = [(load_rollout(p), is_nominal) for p, is_nominal in group]

    # Determine scene features from the first rollout
    first_data = rollouts[0][0]
    has_obstacle = "obstacle_pos" in first_data and "obstacle_half_size" in first_data
    has_target = "target_pos" in first_data

    # Split nominal vs perturbed
    nominal_rollouts = [d for d, nom in rollouts if nom]
    perturbed_rollouts = [d for d, nom in rollouts if not nom]
    n_nominal = len(nominal_rollouts)
    n_perturbed = len(perturbed_rollouts)

    # Compute camera placement from nominal trajectories (all timesteps)
    nom_all_pos = np.concatenate([np.array(d["state"])[:, :3] for d in nominal_rollouts], axis=0)
    centroid = nom_all_pos.mean(axis=0)
    spread = nom_all_pos.max(axis=0) - nom_all_pos.min(axis=0)
    dist = float(max(np.max(spread) * 1.25, 3.0))
    cam_target = centroid + np.array([0.0, 0.0, 0.0])
    cam_pos = centroid + np.array([0.0, -dist, dist * 0.4])

    # Collect trajectory positions for trail rendering
    nominal_pos = [np.array(d["state"])[:, :3] for d in nominal_rollouts]
    perturbed_pos = [np.array(d["state"])[:, :3] for d in perturbed_rollouts]

    # Build model
    xml = build_mjcf(n_nominal, n_perturbed, has_obstacle, has_target, alpha,
                     width=width, height=height, cam_pos=cam_pos, cam_target=cam_target)
    
    # Must match the exact relative paths used in tilthex_laas.xml
    model_assets = {
        "merged/tilthex_laas__26d05b7734899ce35f05c512c442e698_visual.stl": (_ASSETS_DIR / "merged" / "tilthex_laas__26d05b7734899ce35f05c512c442e698_visual.stl").read_bytes(),
        "merged/tilthex_laas__26d05b7734899ce35f05c512c442e698_collision.stl": (_ASSETS_DIR / "merged" / "tilthex_laas__26d05b7734899ce35f05c512c442e698_collision.stl").read_bytes()
    }
    
    model = mujoco.MjModel.from_xml_string(
        xml, assets=model_assets
    )
    data = mujoco.MjData(model)

    # Time stepping: T = number of action steps; state has T+1 entries.
    T_plus_1 = np.array(first_data["state"]).shape[0]
    T = T_plus_1 - 1

    _env_dt = env_dt

    # Frame indices to render
    render_dt = speed / fps
    frame_times = np.arange(0.0, T * _env_dt, render_dt)
    frame_indices = np.clip(
        (frame_times / _env_dt).astype(int), 0, T
    )

    # Set obstacle geometry size once (use first rollout, first timestep)
    if has_obstacle:
        obs_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
        half_size = float(np.array(first_data["obstacle_half_size"])[0, 0])
        model.geom_size[obs_geom_id] = [half_size, half_size, 1.5]

    # Prepare per-rollout state arrays
    def _get_states(rdata):
        state = np.array(rdata["state"])  # (T+1, 13)
        pos = state[:, :3]
        quat = state[:, 3:7]  # [w, x, y, z]
        return pos, quat

    nominal_states = [_get_states(d) for d in nominal_rollouts]
    perturbed_states = [_get_states(d) for d in perturbed_rollouts]

    renderer = mujoco.Renderer(model, height=height, width=width)

    # Validate camera name before entering the loop
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera}' not found in model. "
                         f"Available: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]}")

    obs_mocap_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
                    if has_obstacle else -1)
    tgt_mocap_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
                    if has_target else -1)

    # Pre-parse RGBA colours for trail connectors
    def _parse_rgba(s: str) -> np.ndarray:
        return np.array([float(x) for x in s.split()], dtype=np.float32)

    nom_trail_rgba = _parse_rgba(_DRONE_RGBA_OPAQUE)
    pert_trail_rgba = _parse_rgba(_DRONE_RGBA_TRANSPARENT.format(alpha=max(alpha, 0.15)))

    trail_step_indices = list(range(0, T + 1, _TRAIL_STEP))

    frames = []

    for t in frame_indices:
        if has_obstacle:
            oxy = np.array(first_data["obstacle_pos"])[t]
            data.mocap_pos[model.body_mocapid[obs_mocap_id]] = [oxy[0], oxy[1], 1.5]

        if has_target:
            tgt_pos = np.array(first_data["target_pos"])[t]
            data.mocap_pos[model.body_mocapid[tgt_mocap_id]] = tgt_pos

        for i, (pos, quat) in enumerate(nominal_states):
            _set_drone_qpos(data, model, f"drone_nominal_{i}_joint", pos[t], quat[t])

        for i, (pos, quat) in enumerate(perturbed_states):
            _set_drone_qpos(data, model, f"drone_perturbed_{i}_joint", pos[t], quat[t])

        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera=camera)

        # Draw trail connectors into the scene for all revealed segments
        scene = renderer.scene
        for i, traj in enumerate(nominal_pos):
            for k in range(len(trail_step_indices) - 1):
                if trail_step_indices[k + 1] > t:
                    break
                if scene.ngeom >= scene.maxgeom:
                    break
                p1 = traj[trail_step_indices[k]].astype(np.float64)
                p2 = traj[trail_step_indices[k + 1]].astype(np.float64)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3), np.zeros(9),
                    nom_trail_rgba,
                )
                mujoco.mjv_connector(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    _TRAIL_RADIUS, p1, p2,
                )
                scene.ngeom += 1

        for i, traj in enumerate(perturbed_pos):
            for k in range(len(trail_step_indices) - 1):
                if trail_step_indices[k + 1] > t:
                    break
                if scene.ngeom >= scene.maxgeom:
                    break
                p1 = traj[trail_step_indices[k]].astype(np.float64)
                p2 = traj[trail_step_indices[k + 1]].astype(np.float64)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3), np.zeros(9),
                    pert_trail_rgba,
                )
                mujoco.mjv_connector(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    _TRAIL_RADIUS, p1, p2,
                )
                scene.ngeom += 1

        frames.append(renderer.render().copy())

    renderer.close()
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def write_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    """Write frames (T, H, W, 3) uint8 to an MP4 file using imageio."""
    if not _HAS_IMAGEIO:
        raise ImportError("imageio is required to write video files. Install it with: pip install imageio[ffmpeg]")
    try:
        writer = imageio.get_writer(str(output_path), fps=fps, format="FFMPEG",
                                    codec="libx264", quality=7, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        # Fallback: try imageio v3 API
        imageio.mimwrite(str(output_path), frames, fps=fps, format="FFMPEG",
                         codec="libx264", quality=7)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Animate drone rollout trajectories using MuJoCo renderer."
    )
    p.add_argument(
        "rollout_dir",
        type=Path,
        help="Directory containing traj_*.msgpack rollout files.",
    )
    p.add_argument(
        "--traj",
        type=int,
        default=None,
        metavar="N",
        help="Only render trajectory group N. If omitted, all groups are rendered.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for MP4 files. Defaults to the rollout directory.",
    )
    p.add_argument(
        "--camera",
        choices=["track_side", "top_down"],
        default="track_side",
        help="Camera to use for rendering (default: track_side).",
    )
    p.add_argument("--fps",   type=int,   default=30,  help="Output video fps (default: 30).")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0).")
    p.add_argument("--width", type=int,   default=1920, help="Frame width in pixels (default: 1280).")
    p.add_argument("--height",type=int,   default=1080,  help="Frame height in pixels (default: 720).")
    p.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation timestep in seconds (default: 0.02 = 50 Hz).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Transparency of perturbed drone bodies (0=invisible, 1=opaque; default: 0.1).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    rollout_dir: Path = args.rollout_dir.resolve()
    if not rollout_dir.is_dir():
        print(f"Error: {rollout_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output.resolve() if args.output else rollout_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _MESH_PATH.exists():
        print(f"Error: mesh not found at {_MESH_PATH}", file=sys.stderr)
        sys.exit(1)

    groups = group_rollout_files(rollout_dir)
    if not groups:
        print(f"No traj_*.msgpack files found in {rollout_dir}.", file=sys.stderr)
        sys.exit(1)

    traj_ids = sorted(groups.keys())
    if args.traj is not None:
        if args.traj not in groups:
            print(f"Trajectory {args.traj} not found. Available: {traj_ids}", file=sys.stderr)
            sys.exit(1)
        traj_ids = [args.traj]

    # Flatten all selected groups into a single scene
    combined: List[Tuple[Path, bool]] = []
    for traj_id in traj_ids:
        combined.extend(groups[traj_id])

    n_nominal = sum(1 for _, nom in combined if nom)
    n_perturbed = sum(1 for _, nom in combined if not nom)
    print(
        f"[animate] {len(traj_ids)} trajectory group(s): "
        f"{n_nominal} nominal, {n_perturbed} perturbed ({len(combined)} total)"
    )

    try:
        frames = render_trajectory_group(
            combined,
            camera=args.camera,
            fps=args.fps,
            speed=args.speed,
            width=args.width,
            height=args.height,
            alpha=args.alpha,
            env_dt=args.dt,
        )
    except Exception as e:
        print(f"  Error rendering: {e}", file=sys.stderr)
        sys.exit(1)

    label = f"traj_{args.traj:05d}" if args.traj is not None else "all"
    out_path = output_dir / f"{label}_{args.camera}.mp4"
    print(f"  Writing {len(frames)} frames → {out_path}")
    try:
        write_video(frames, out_path, fps=args.fps)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Error writing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
