#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scene.dataset_readers import sceneLoadTypeCallbacks, storePly


def load_scene_info(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](
            args.source_path,
            args.images,
            args.depths,
            args.eval,
            args.train_test_exp,
        )
    if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        return sceneLoadTypeCallbacks["Blender"](
            args.source_path,
            args.white_background,
            args.depths,
            args.eval,
        )
    raise RuntimeError("Scene type not recognized. Expected Colmap or Blender.")


def main():
    parser = argparse.ArgumentParser(description="Normalize point cloud to camera normalization (translate+scale).")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--depths", default="", type=str)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--train_test_exp", action="store_true", default=False)
    parser.add_argument("--white_background", action="store_true", default=False)
    parser.add_argument("--output", default=None, type=str, help="输出 PLY 路径，默认写入 <source_path>/sparse/0/points3D_normalized.ply")
    args = parser.parse_args()

    scene_info = load_scene_info(args)
    if scene_info.point_cloud is None:
        print("Point cloud not found. Cannot normalize.")
        return 1

    points = np.asarray(scene_info.point_cloud.points).astype(np.float64)
    colors = np.asarray(scene_info.point_cloud.colors).astype(np.float32)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    diag = np.linalg.norm(bbox_max - bbox_min)

    cam_center = -scene_info.nerf_normalization["translate"]
    cam_radius = scene_info.nerf_normalization["radius"]

    print("Original bbox_center:", center)
    print("Original diag:", diag)
    print("Camera center:", cam_center)
    print("Camera radius:", cam_radius)

    if diag <= 0:
        print("Invalid point cloud diag; aborting.")
        return 1

    scale = (2.0 * cam_radius) / diag
    new_points = (points - center) * scale + cam_center

    if args.output is None:
        out_ply = os.path.join(args.source_path, "sparse", "0", "points3D_normalized.ply")
    else:
        out_ply = args.output

    out_dir = os.path.dirname(out_ply)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # storePly expects rgb in 0-255 int? original storePly used rgb as values [0-1]? In dataset_readers.storePly it concatenates rgb as-is and writes u1
    # Prepare rgb as uint8
    rgb_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    storePly(out_ply, new_points, rgb_uint8)

    print(f"Wrote normalized point cloud to: {out_ply}")
    print(f"Applied scale={scale:.6g}, translated center->{cam_center}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
