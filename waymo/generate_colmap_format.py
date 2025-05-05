import os
import dataclasses
import numpy as np
from typing import List, Tuple
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion

from .dataset import CAM_LABELS, WaymoData, CamData, LidarData, Pose


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
            fx=cam_data.intrinsic[0],  # fx
            fy=cam_data.intrinsic[1],  # fy
            cx=cam_data.intrinsic[2],  # cx
            cy=cam_data.intrinsic[3],  # cy
        )


opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

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
        # camera to vehicle
        mat = np.linalg.inv(cam_data.extrinsic @ opencv2camera)
        # project to origin coordinate
        cam_data_pose = to_origin(Pose.from_matrix(mat), ego_pose, origin_pose)
        # rotate to convert coordinate system
        q = Quaternion(matrix=WORLD2CAM @ Quaternion(cam_data_pose.rotation).rotation_matrix).inverse
        t = -q.rotation_matrix @ WORLD2CAM @ cam_data_pose.translation
        path = Path("/tmp/waymo_images")
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"{idx}.jpg"
        cam_data.data.save(path)
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
            name="dummy",
            points2d=[],
            data=cam_data.data,
            path=path,
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
    def from_points(cls, idx: int, point: np.ndarray, ego_pose: Pose, origin_pose: Pose) -> "ColmapPoint3D":
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


def get_colmap_data(data: WaymoData, origin_pose: Pose) -> Tuple[List[Camera], List[ColmapImage], List[ColmapPoint3D]]:
    global CAMERA_COUNTER
    global IMAGE_COUNTER
    global POINT_COUNTER
    ego_pose = Pose.from_matrix(data.lidar_data.ego_pose)

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
    ego_points = data.lidar_data.to_ego_origin().T
    for i in range(data.lidar_data.data.shape[0]):
        points.append(ColmapPoint3D.from_points(POINT_COUNTER + 1, ego_points[:3, i], ego_pose, origin_pose))
        POINT_COUNTER += 1
    unused_points = set(range(len(points)))

    # project LiDAR to pixel 2D plane
    for image_idx, label in enumerate(CAM_LABELS):
        cam_data = data.cam_sensors[label]
        image = images[image_idx]
        tpoints, mask = data.lidar_data.to_img_coord(cam_data)
        for point_idx, inframe in enumerate(mask):
            point = points[point_idx]
            if not inframe:
                continue
            image.points2d.append((tpoints[point_idx, 0], tpoints[point_idx, 1], point.id))
            r, g, b = cam_data.data.getpixel((int(tpoints[point_idx, 0]), int(tpoints[point_idx, 1])))
            point.r = r
            point.g = g
            point.b = b
            point.tracks.append((image.id, len(image.points2d) - 1))

            # remove point_idx from unused_points
            if point_idx in unused_points:
                unused_points.remove(point_idx)

    # remove unused points
    points = [points[point_idx] for point_idx in range(len(points)) if point_idx not in unused_points]

    return cameras, images, points
