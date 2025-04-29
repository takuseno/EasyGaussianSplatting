import dataclasses
import copy
from typing import Tuple, Dict

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pathlib import Path
from PIL import Image
from pyquaternion import Quaternion
import numpy as np


ROOT_DIR = Path('/home/takuma/datasets/nuscenes')
CAM_LABELS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]


@dataclasses.dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation: np.ndarray


@dataclasses.dataclass(frozen=True)
class CamData:
    data: Image.Image
    ego_pose: Pose
    calibration: Pose
    camera_intrinsic: np.ndarray
    filename: str
    path: Path



@dataclasses.dataclass(frozen=True)
class LidarData:
    data: LidarPointCloud
    ego_pose: Pose
    calibration: Pose

    def to_img_coord(self, cam_data: CamData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = copy.deepcopy(self.data)

        # calibration data
        # step 1: from lidar to ego vehicle
        data.rotate(Quaternion(self.calibration.rotation).rotation_matrix)
        data.translate(self.calibration.translation)

        # step 2: from ego to global
        data.rotate(Quaternion(self.ego_pose.rotation).rotation_matrix)
        data.translate(self.ego_pose.translation)

        # step 3: from global to ego
        data.translate(-cam_data.ego_pose.translation)
        data.rotate(Quaternion(cam_data.ego_pose.rotation).rotation_matrix.T)

        # step 4: from ego to camera
        data.translate(-cam_data.calibration.translation)
        data.rotate(Quaternion(cam_data.calibration.rotation).rotation_matrix.T)

        # translate point clouds to image plane
        points = data.points[:3, :]
        depths = points[2, :]
        view = cam_data.camera_intrinsic
        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view
        nbr_points = points.shape[1]
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        # remove points outside or behind camera view
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 1)  # remove too close points
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < cam_data.data.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < cam_data.data.size[1] - 1)

        return points, mask, depths

    def to_ego_origin(self) -> np.ndarray:
        data = copy.deepcopy(self.data)
        # calibration data
        # step 1: from lidar to ego vehicle
        data.rotate(Quaternion(self.calibration.rotation).rotation_matrix)
        data.translate(self.calibration.translation)
        return data.points


class NuSceneData:
    def __init__(self, idx: int):
        # load NuScene data point
        nusc = NuScenes(version='v1.0-mini', dataroot=str(ROOT_DIR), verbose=True)
        scene = nusc.scene[0]
        first_sample_token = scene["first_sample_token"]
        sample = nusc.get("sample", first_sample_token)
        for _ in range(idx):
            sample = nusc.get("sample", sample["next"])

        # load camera data
        self.cam_sensors: Dict[str, CamData] = {}
        for label in CAM_LABELS:
            cam_data = nusc.get("sample_data", sample["data"][label])
            image_path = ROOT_DIR / cam_data["filename"]
            image = Image.open(str(image_path))
            ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
            cam_calibration = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            self.cam_sensors[label] = CamData(
                data=image,
                ego_pose=Pose(
                    translation=np.array(ego_pose["translation"]),
                    rotation=np.array(ego_pose["rotation"]),
                ),
                calibration=Pose(
                    translation=np.array(cam_calibration["translation"]),
                    rotation=np.array(cam_calibration["rotation"]),
                ),
                camera_intrinsic=np.array(cam_calibration["camera_intrinsic"]),
                filename=cam_data["filename"],
                path=image_path,
            )

        # load lidar data
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(str(ROOT_DIR / lidar_data["filename"]))
        lidar_calibration = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        self.lidar_data = LidarData(
            data=pc,
            ego_pose=Pose(
                translation=np.array(ego_pose["translation"]),
                rotation=np.array(ego_pose["rotation"]),
            ),
            calibration=Pose(
                translation=np.array(lidar_calibration["translation"]),
                rotation=np.array(lidar_calibration["rotation"]),
            ),
        )
