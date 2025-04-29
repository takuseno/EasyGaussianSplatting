import os
import dataclasses
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion

from .dataset import CAM_LABELS, NuSceneData, Pose, CamData


CAM2WORLD = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
WORLD2CAM = np.linalg.inv(CAM2WORLD)

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

    @classmethod
    def from_cam_data(cls, idx: int, cam_data: CamData) -> "Camera":
        return Camera(
            id=idx,
            model="PINHOLE",
            width=cam_data.data.size[0],
            height=cam_data.data.size[1],
            fx=cam_data.camera_intrinsic[0][0],  # fx
            fy=cam_data.camera_intrinsic[1][1],  # fy
            cx=cam_data.camera_intrinsic[0][2],  # cx
            cy=cam_data.camera_intrinsic[1][2],  # cy
        )


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

    @classmethod
    def from_cam_data(cls, idx: int, cam_data: CamData, ego_pose: Pose, origin_pose: Pose) -> "ColmapImage":
        # project to origin coordinate
        cam_data_pose = to_origin(cam_data.calibration, ego_pose, origin_pose)
        # rotate to convert coordinate system
        q = Quaternion(matrix=WORLD2CAM @ Quaternion(cam_data_pose.rotation).rotation_matrix).inverse
        t = -q.rotation_matrix @ WORLD2CAM @ cam_data_pose.translation
        return ColmapImage(
            id=idx,
            qw=q.w,
            qx=q.x,
            qy=q.y,
            qz=q.z,
            tx=t[0],
            ty=t[1],
            tz=t[2],
            camera_id=idx,
            name=os.path.basename(cam_data.filename),
            points2d=[],
            data=cam_data.data,
            path=cam_data.path,
        )


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

    @classmethod
    def from_points(cls, idx: int, point: np.ndarray, calibration: Pose, ego_pose: Pose, origin_pose: Pose) -> "ColmapPoint3D":
        # project to origin coordinate
        origin_rotmat_inv = Quaternion(origin_pose.rotation).inverse.rotation_matrix
        ego_rotmat = Quaternion(ego_pose.rotation).rotation_matrix
        offset_rotmat = origin_rotmat_inv @ ego_rotmat
        offset_translation = origin_rotmat_inv @ (ego_pose.translation - origin_pose.translation)
        # rotate to convert coordinate system
        x, y, z = WORLD2CAM @ ((offset_rotmat @ point) + offset_translation)
        return ColmapPoint3D(
            id=idx,
            x=x,
            y=y,
            z=z,
            r=0,
            g=0,
            b=0,
            error=0,
            tracks=[],
        )


def to_origin(pose: Pose, ego_pose: Pose, origin_pose: Pose) -> Pose:
    origin_rotmat_inv = Quaternion(origin_pose.rotation).inverse.rotation_matrix
    ego_rotmat = Quaternion(ego_pose.rotation).rotation_matrix
    pose_rotmat = Quaternion(pose.rotation).rotation_matrix
    offset_rotmat = Quaternion(matrix=origin_rotmat_inv @ ego_rotmat).rotation_matrix
    offset_translation = origin_rotmat_inv @ (ego_pose.translation - origin_pose.translation)
    new_translation = pose.translation + offset_translation
    new_rot = Quaternion(matrix=offset_rotmat @ pose_rotmat)
    return Pose(
        translation=new_translation,
        rotation=np.array([new_rot.w, new_rot.x, new_rot.y, new_rot.z]),
    )

CAMERA_COUNTER = 0
IMAGE_COUNTER = 0
POINT_COUNTER = 0


def get_colmap_data(data: NuSceneData, origin_pose: Pose) -> Tuple[List[Camera], List[ColmapImage], List[ColmapPoint3D]]:
    global CAMERA_COUNTER
    global IMAGE_COUNTER
    global POINT_COUNTER
    ego_pose = data.lidar_data.ego_pose

    # make cameras.txt
    cameras = []
    for i, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        cameras.append(Camera.from_cam_data(CAMERA_COUNTER + 1, cam_data))
        CAMERA_COUNTER += 1

    images: List[ColmapImage] = []
    for i, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        images.append(ColmapImage.from_cam_data(IMAGE_COUNTER + 1, cam_data, ego_pose, origin_pose))
        IMAGE_COUNTER += 1

    points: List[ColmapPoint3D] = []
    ego_points = data.lidar_data.to_ego_origin()
    for i in range(data.lidar_data.data.points.shape[1]):
        points.append(ColmapPoint3D.from_points(POINT_COUNTER + 1, ego_points[:3, i], data.lidar_data.calibration, ego_pose, origin_pose))
        POINT_COUNTER += 1

    # project LiDAR to pixel 2D plane
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
