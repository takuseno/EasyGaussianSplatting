#!/usr/bin/env python3

import numpy as np
from pyquaternion import Quaternion
from gsplat.gau_io import rotate_gaussian, load_gs, get_example_gs
from gsplat.gausplat_dataset import NuSceneGSplatDataset, GSplatDataset, WaymoGSplatDataset
from nuscene.dataset import Pose

from PyQt5.QtWidgets import QApplication
from viewer.viewer import Viewer
from viewer.custom_items import GLCameraFrameItem, GaussianItem, GridItem


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the trained gs data")
    parser.add_argument("--path", help="the path of dataset")
    parser.add_argument("--nuscene", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    parser.add_argument("--skip", help="skip", default=5)
    parser.add_argument("--idx", type=int, default=1)
    args = parser.parse_args()
    cam_2_world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    gs_set = []
    cam_size = 1
    if args.nuscene:
        gs_set = NuSceneGSplatDataset(idx=args.idx)
        gs = gs_set.gs
        cam_size = gs_set.sence_size * 0.05
        rotate_gaussian(cam_2_world, gs)
    elif args.waymo:
        gs_set = WaymoGSplatDataset(idx=args.idx)
        gs = gs_set.gs
        cam_size = gs_set.sence_size# * 0.05
        rotate_gaussian(cam_2_world, gs)
    elif args.path:
        print("Try to training %s ..." % args.path)
        gs_set = GSplatDataset(args.path)
        gs = gs_set.gs
        cam_size = gs_set.sence_size * 0.05
        rotate_gaussian(cam_2_world, gs)
    elif args.gs:
        print("Try to load %s ..." % args.gs)
        gs_set = NuSceneGSplatDataset(idx=args.idx)
        gs = load_gs(args.gs)
        rotate_gaussian(cam_2_world, gs)

    if (not args.gs) and (not args.path) and (not args.nuscene) and (not args.waymo):
        print("not gs file.")
        gs = get_example_gs()

    gs_data = gs.view(np.float32).reshape(gs.shape[0], -1)

    app = QApplication([])
    gs_item = GaussianItem()
    grid_item = GridItem()

    items = [("grid", grid_item), ("gs", gs_item)]

    for i in range(len(gs_set)):
        #if (i % args.skip != 0):
        #    continue
        cam, _ = gs_set[i]
        T = np.eye(4)
        Rcw = cam.Rcw.cpu().numpy()
        tcw = cam.tcw.cpu().numpy()
        Rwc = np.linalg.inv(Rcw)
        twc = Rwc @ (-tcw)
        T[:3, :3] = cam_2_world @ Rwc
        T[:3, 3] = cam_2_world @ twc
        cam_item = GLCameraFrameItem(T=T, path=cam.path, size=cam_size)
        items.append(("cam%04d"%cam.id, cam_item))

    viewer = Viewer(dict(items))
    viewer.items["gs"].setData(gs_data=gs_data)
    viewer.show()
    app.exec_()
