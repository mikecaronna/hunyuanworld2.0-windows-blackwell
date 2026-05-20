"""
Rotate a 3DGS PLY file (positions + per-Gaussian quaternions).

Usage:
    python rotate_3dgs.py <input.ply> <output.ply> --axis X|Y|Z --angle <degrees>

Common fixes:
    --axis X --angle -90    Z-up to Y-up
    --axis X --angle 180    COLMAP / OpenCV (Y-down) to Y-up
    --axis Z --angle -90    Roll the scene clockwise 90° around forward axis
    --axis Z --angle 90     Roll the scene counter-clockwise 90°
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def quat_mul(q1, q2):
    """Hamilton product of quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def axis_angle_to_quat(axis, angle_rad):
    half = angle_rad / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def axis_angle_to_matrix(axis, angle_rad):
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--axis", choices=["X", "Y", "Z"], required=True)
    p.add_argument("--angle", type=float, required=True, help="degrees")
    args = p.parse_args()

    axis_vec = {"X": np.array([1, 0, 0]), "Y": np.array([0, 1, 0]), "Z": np.array([0, 0, 1])}[args.axis]
    angle_rad = np.deg2rad(args.angle)
    R = axis_angle_to_matrix(axis_vec, angle_rad)
    Rq = axis_angle_to_quat(axis_vec, angle_rad)

    print(f"[Load] {args.input}")
    ply = PlyData.read(args.input)
    elem = ply["vertex"]
    n = len(elem)
    print(f"[Load] {n} Gaussians")

    # Rotate positions
    xyz = np.stack([np.asarray(elem["x"]), np.asarray(elem["y"]), np.asarray(elem["z"])], axis=1)
    xyz_rot = xyz @ R.T
    elem["x"][:] = xyz_rot[:, 0].astype(elem["x"].dtype)
    elem["y"][:] = xyz_rot[:, 1].astype(elem["y"].dtype)
    elem["z"][:] = xyz_rot[:, 2].astype(elem["z"].dtype)
    print(f"[Rotate] positions: axis={args.axis} angle={args.angle}deg")

    # Rotate per-Gaussian orientations (stored as rot_0..rot_3 = w, x, y, z)
    rot_field_names = [name for name in elem.data.dtype.names if name.startswith("rot_")]
    if len(rot_field_names) >= 4:
        rot_field_names = sorted(rot_field_names, key=lambda s: int(s.split("_")[1]))[:4]
        quats = np.stack([np.asarray(elem[name]) for name in rot_field_names], axis=1)
        # Apply Rq composed with each gaussian's quat: q_new = Rq * q
        Rq_tile = np.broadcast_to(Rq, quats.shape)
        quats_new = quat_mul(Rq_tile, quats)
        # Renormalize
        norms = np.linalg.norm(quats_new, axis=1, keepdims=True)
        quats_new = quats_new / np.maximum(norms, 1e-10)
        for i, name in enumerate(rot_field_names):
            elem[name][:] = quats_new[:, i].astype(elem[name].dtype)
        print(f"[Rotate] quaternions: {len(rot_field_names)} fields {rot_field_names}")
    else:
        print(f"[Skip] no rot_0..rot_3 fields found (got: {rot_field_names})")

    # Note: spherical harmonics (f_rest_*) are NOT rotated here. View-dependent
    # color may look slightly off at oblique angles. For a basic visualization
    # fix this is fine — most viewers handle SH gracefully.

    print(f"[Write] {args.output}")
    PlyData([elem], text=ply.text).write(args.output)
    print(f"[Done] {args.output}")


if __name__ == "__main__":
    main()
