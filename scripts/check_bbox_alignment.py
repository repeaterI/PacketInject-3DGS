#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scene.dataset_readers import sceneLoadTypeCallbacks


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
    parser = argparse.ArgumentParser(description="Check bbox vs camera normalization.")
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--images", default="images", type=str)
    parser.add_argument("--depths", default="", type=str)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--train_test_exp", action="store_true", default=False)
    parser.add_argument("--white_background", action="store_true", default=False)
    args = parser.parse_args()

    scene_info = load_scene_info(args)
    if scene_info.point_cloud is None:
        print("Point cloud not found. Cannot compute bbox.")
        return 1

    points = np.asarray(scene_info.point_cloud.points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    diag = np.linalg.norm(bbox_max - bbox_min)

    cam_center = -scene_info.nerf_normalization["translate"]
    cam_radius = scene_info.nerf_normalization["radius"]
    center_offset = np.linalg.norm(center - cam_center)
    diag_to_radius = diag / max(cam_radius * 2.0, 1e-6)

    print("bbox_min:", bbox_min)
    print("bbox_max:", bbox_max)
    print("bbox_center:", center)
    print("bbox_diag:", diag)
    print("camera_center:", cam_center)
    print("camera_radius:", cam_radius)
    print("center_offset:", center_offset)
    print("diag_to_camera_diameter:", diag_to_radius)

    if center_offset > cam_radius * 0.5:
        print("WARNING: point cloud center is far from camera center. Check for extra transforms.")
    if diag_to_radius < 0.2 or diag_to_radius > 5.0:
        print("WARNING: point cloud scale and camera scale differ a lot. Check normalization.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
