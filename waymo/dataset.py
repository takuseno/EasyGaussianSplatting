import dataclasses
import io
from typing import Dict, Tuple
from PIL import Image

from pyquaternion import Quaternion
import cv2
import numpy as np

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils


CAM_LABELS = [
    dataset_pb2.CameraName.FRONT,
    dataset_pb2.CameraName.FRONT_LEFT,
    dataset_pb2.CameraName.FRONT_RIGHT,
    dataset_pb2.CameraName.SIDE_LEFT,
    dataset_pb2.CameraName.SIDE_RIGHT,
]


def euler_to_rotation_matrix(yaw, pitch, roll):
    # Assumes input in radians
    cy, cp, cr = np.cos([yaw, pitch, roll])
    sy, sp, sr = np.sin([yaw, pitch, roll])

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    
    Ry = np.array([
        [cp, 0, sp],
        [ 0, 1,  0],
        [-sp, 0, cp]
    ])
    
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    
    R = Rz @ Ry @ Rx  # Note: ZYX order
    return R


@dataclasses.dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "Pose":
        #mat = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) @ mat
        quaternion = Quaternion(matrix=mat[:3, :3])
        translation = mat[:3, 3].reshape(-1)
        #translation = np.array([-translation[1], -translation[2], translation[0]])
        #offset_rot = euler_to_rotation_matrix(np.pi / 2, np.pi / 2, 0)
        #offset_rot = euler_to_rotation_matrix(0, np.pi, np.pi) @ offset_rot
        #quaternion = Quaternion(np.array([quaternion.w, -quaternion.y, -quaternion.z, quaternion.x]))
        #quaternion = Quaternion(matrix=offset_rot @ quaternion.rotation_matrix)
        #return Pose(np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z]), offset_rot @ translation)
        #translation = np.array([-translation[1], -translation[2], translation[0]])
        return Pose(np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z]), translation)


@dataclasses.dataclass(frozen=True)
class CamData:
    data: Image.Image
    ego_pose: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    vehicle_to_image: np.ndarray


@dataclasses.dataclass(frozen=True)
class LidarData:
    data: np.ndarray
    ego_pose: np.ndarray
    extrinsic: np.ndarray

    def to_img_coord(self, cam_data: CamData) -> Tuple[np.ndarray, np.ndarray]:
        img = cam_data.data
        pcl1 = np.concatenate((self.data, np.ones_like(self.data[:,0:1])),axis=1)
        proj_pcl_3d = np.einsum('ij,bj->bi', cam_data.vehicle_to_image, pcl1)

        # Project the point cloud onto the image.
        proj_pcl = proj_pcl_3d[:,:2] / proj_pcl_3d[:,2:3]

        # Filter LIDAR points which are behind the camera.
        # Filter points which are outside the image.
        mask = np.logical_and(
            np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.width),
            np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.height),
        )
        mask = np.logical_and(mask, proj_pcl_3d[:,2] > 0)

        return proj_pcl, mask

    def to_ego_origin(self) -> np.ndarray:
        return self.data
        #pcl1 = np.concatenate((self.data, np.ones_like(self.data[:,0:1])),axis=1)
        #arr = (self.extrinsic @ pcl1.T).T[:, :3]
        #return arr


class WaymoData:
    def __init__(self):
        self.datafile = WaymoDataFileReader("/home/takuma/datasets/waymo/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord")
        # Generate a table of the offset of all frame records in the file.
        table = self.datafile.get_record_table()

        frame = next(self.datafile)
        assert frame

        ego_pose = np.array(frame.pose.transform).reshape(4, 4)

        # extract camera data
        self.cam_sensors: Dict[dataset_pb2.CameraName, CamData]  = {}
        for label in CAM_LABELS:
            camera_calibration = utils.get(frame.context.camera_calibrations, label)
            camera = utils.get(frame.images, label)
            img = Image.open(io.BytesIO(camera.image))
            cam_data = CamData(
                data=img,
                ego_pose=ego_pose,
                extrinsic=np.array(camera_calibration.extrinsic.transform).reshape(4, 4),
                intrinsic=np.array(camera_calibration.intrinsic),
                vehicle_to_image=utils.get_image_transform(camera_calibration),
            )
            self.cam_sensors[label] = cam_data

        # extract lidar data
        laser = utils.get(frame.lasers, dataset_pb2.LaserName.TOP)
        laser_calibration = utils.get(frame.context.laser_calibrations, dataset_pb2.LaserName.TOP)
        ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
        pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
        self.lidar_data = LidarData(
            data=pcl,
            ego_pose=ego_pose,
            extrinsic=np.array(laser_calibration.extrinsic.transform).reshape(4, 4),
        )
