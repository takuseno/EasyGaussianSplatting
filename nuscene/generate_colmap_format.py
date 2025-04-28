import os
import dataclasses
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion

from .dataset import CAM_LABELS, NuSceneData

root_dir = Path(os.path.dirname(__file__))
(root_dir / "images").mkdir(exist_ok=True)
(root_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclasses.dataclass(frozen=True)
class ColmapImage:
    id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Tuple[int, int, int]]
    data: Image.Image
    path: Path


@dataclasses.dataclass
class ColmapPoint3D:
    id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float
    tracks: List[Tuple[int, int]]


def get_colmap_data() -> Tuple[List[Camera], List[ColmapImage], List[ColmapPoint3D]]:
    data = NuSceneData()
    cam2world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    world2cam = np.linalg.inv(cam2world)

    # make cameras.txt
    cameras = []
    for i, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        camera = Camera(
            id=i + 1,
            model="PINHOLE",
            width=cam_data.data.size[0],
            height=cam_data.data.size[1],
            fx=cam_data.camera_intrinsic[0][0],  # fx
            fy=cam_data.camera_intrinsic[1][1],  # fy
            cx=cam_data.camera_intrinsic[0][2],  # cx
            cy=cam_data.camera_intrinsic[1][2],  # cy
        )
        cameras.append(camera)

    images: List[ColmapImage] = []
    for i, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        # rotate to convert coordinate system
        q = Quaternion(matrix=world2cam @ Quaternion(cam_data.calibration.rotation).rotation_matrix).inverse
        t = -q.rotation_matrix @ world2cam @ cam_data.calibration.translation
        image = ColmapImage(
            id=i + 1,
            qw=q.w,
            qx=q.x,
            qy=q.y,
            qz=q.z,
            tx=t[0],
            ty=t[1],
            tz=t[2],
            camera_id=i + 1,
            name=os.path.basename(cam_data.filename),
            points2d=[],
            data=cam_data.data,
            path=cam_data.path,
        )
        images.append(image)

    points: List[ColmapPoint3D] = []
    ego_points = data.lidar_data.to_ego_origin()
    for i in range(data.lidar_data.data.points.shape[1]):
        # rotate to convert coordinate system
        x, y, z = world2cam @ ego_points[:3, i]
        point = ColmapPoint3D(
            id=i + 1,
            x=x,
            y=y,
            z=z,
            r=0,
            g=0,
            b=0,
            error=0,
            tracks=[],
        )
        points.append(point)

    for image_idx, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        image = images[image_idx]
        tpoints, mask, depths = data.lidar_data.to_img_coord(cam_data)
        for point_idx, inframe in enumerate(mask):
            point = points[point_idx]
            if not inframe:
                continue
            image.points2d.append((tpoints[0][point_idx], tpoints[1][point_idx], point.id))
            r, g, b = cam_data.data.getpixel((int(tpoints[0][point_idx]), int(tpoints[1][point_idx])))
            point.r = r
            point.g = g
            point.b = b
            point.tracks.append((image.id, len(image.points2d) - 1))

    return cameras, images, points
