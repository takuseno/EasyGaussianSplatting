from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from gsplat.read_write_model import read_model, read_points_bin_as_gau, qvec2rotmat, SH_C0_0
# from read_write_model import read_write_model, 
from PIL import Image
import torch
import torchvision
from plyfile import PlyData
import torchvision.transforms as transforms
import faiss
import numpy as np
from nuscene.dataset import NuSceneData
from nuscene.generate_colmap_format import get_colmap_data
from waymo.dataset import WaymoData, Pose as WaymoPose
from waymo.generate_colmap_format import get_colmap_data as get_waymo_colmap_data


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw, path):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw
        self.path = path



class GSplatDataset(Dataset):
    def __init__(self, path, resize_rate=1, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.resize_rate = resize_rate

        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')
        self.cameras = []
        self.images = []
        for image_param in image_params.values():
            i = image_param.camera_id
            camera_param = camera_params[i]
            im_path = str(Path(path, "images", image_param.name))
            image = Image.open(im_path)
            if (resize_rate != 1):
                image = image.resize((image.width * self.resize_rate, image.height * self.resize_rate))

            w_scale = image.width/camera_param.width
            h_scale = image.height/camera_param.height
            fx = camera_param.params[0] * w_scale
            fy = camera_param.params[1] * h_scale
            cx = camera_param.params[2] * w_scale
            cy = camera_param.params[3] * h_scale
            Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(image_param.tvec).to(self.device).to(torch.float32)
            camera = Camera(image_param.id, image.width, image.height, fx, fy, cx, cy, Rcw, tcw, im_path)
            image = torchvision.transforms.functional.to_tensor(image).to(self.device).to(torch.float32)

            self.cameras.append(camera)
            self.images.append(image)
        try:
            self.gs = np.load(Path(path, "sparse/0/points3D.npy"))
        except:
            self.gs = read_points_bin_as_gau(Path(path, "sparse/0/points3D.bin"))
            np.save(Path(path, "sparse/0/points3D.npy"), self.gs)

        twcs = torch.stack([x.twc for x in self.cameras])
        cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
        self.sence_size = float(torch.max(cam_dist)) * 1.1

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)


class NuSceneGSplatDataset(Dataset):
    def __init__(self, idx: int, resize_rate: int = 1, device: str = "cuda"):
        self.resize_rate = resize_rate
        self.device = device

        origin_pose = NuSceneData(0).lidar_data.ego_pose
        nuscene_cameras = []
        nuscene_images = []
        nuscene_points = []
        for idx in range(3):
            data = NuSceneData(idx)
            colmap_data = get_colmap_data(data, origin_pose)
            nuscene_cameras.extend(colmap_data[0])
            nuscene_images.extend(colmap_data[1])
            nuscene_points.extend(colmap_data[2])

        self.cameras = []
        self.images = []
        for cam, image in zip(nuscene_cameras, nuscene_images):
            im_data = image.data
            if (resize_rate != 1):
                im_data = im_data.resize((im_data.width * self.resize_rate, im_data.height * self.resize_rate))

            w_scale = im_data.width / cam.width
            h_scale = im_data.height / cam.height
            fx = cam.fx * w_scale
            fy = cam.fy * h_scale
            cx = cam.cx * w_scale
            cy = cam.cy * h_scale
            Rcw = torch.from_numpy(qvec2rotmat((image.qw, image.qx, image.qy, image.qz))).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(np.array([image.tx, image.ty, image.tz])).to(self.device).to(torch.float32)
            camera = Camera(cam.id, cam.width, cam.height, fx, fy, cx, cy, Rcw, tcw, image.path)
            torch_image = torchvision.transforms.functional.to_tensor(im_data).to(self.device).to(torch.float32)

            self.cameras.append(camera)
            self.images.append(torch_image)

        num_points = len(nuscene_points)
        pws = np.zeros((num_points, 3))
        shs = np.zeros((num_points, 3))
        for i, point in enumerate(nuscene_points):
            pws[i] = np.array([point.x, point.y, point.z])
            shs[i] = (np.array([point.r, point.g, point.b]) / 255 - 0.5) / SH_C0_0
        rots = np.zeros([num_points, 4])
        rots[:, 0] = 1
        alphas = np.ones([num_points]) * 0.8
        pws = pws.astype(np.float32)
        rots = rots.astype(np.float32)
        alphas = alphas.astype(np.float32)
        shs = shs.astype(np.float32)

        N, D = pws.shape
        index = faiss.IndexFlatL2(D)
        index.add(pws)
        distances, indices = index.search(pws, 2)
        distances = np.clip(distances[:, 1], 0.01, 3)
        scales = distances[:, np.newaxis].repeat(3, 1)

        dtypes = [('pw', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('alpha', '<f4'),
                  ('sh', '<f4', (3,))]

        self.gs = np.rec.fromarrays([pws, rots, scales, alphas, shs], dtype=dtypes)

        twcs = torch.stack([x.twc for x in self.cameras])
        cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
        self.sence_size = float(torch.max(cam_dist)) * 1.1

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)


class WaymoGSplatDataset(Dataset):
    def __init__(self, idx: int, resize_rate: int = 1, device: str = "cuda"):
        self.resize_rate = resize_rate
        self.device = device

        origin_pose = WaymoPose.from_matrix(WaymoData(0).lidar_data.ego_pose)
        nuscene_cameras = []
        nuscene_images = []
        nuscene_points = []

        for idx in range(5):
            data = WaymoData(idx)
            colmap_data = get_waymo_colmap_data(data, origin_pose)
            nuscene_cameras.extend(colmap_data[0])
            nuscene_images.extend(colmap_data[1])
            nuscene_points.extend(colmap_data[2])

        self.cameras = []
        self.images = []
        for cam, image in zip(nuscene_cameras, nuscene_images):
            im_data = image.data
            if (resize_rate != 1):
                im_data = im_data.resize((im_data.width * self.resize_rate, im_data.height * self.resize_rate))

            w_scale = im_data.width / cam.width
            h_scale = im_data.height / cam.height
            fx = cam.fx * w_scale
            fy = cam.fy * h_scale
            cx = cam.cx * w_scale
            cy = cam.cy * h_scale
            Rcw = torch.from_numpy(qvec2rotmat((image.qw, image.qx, image.qy, image.qz))).to(self.device).to(torch.float32)
            tcw = torch.from_numpy(np.array([image.tx, image.ty, image.tz])).to(self.device).to(torch.float32)
            camera = Camera(cam.id, cam.width, cam.height, fx, fy, cx, cy, Rcw, tcw, image.path)
            torch_image = torchvision.transforms.functional.to_tensor(im_data).to(self.device).to(torch.float32)

            self.cameras.append(camera)
            self.images.append(torch_image)

        num_points = len(nuscene_points)
        pws = np.zeros((num_points, 3))
        shs = np.zeros((num_points, 3))
        for i, point in enumerate(nuscene_points):
            pws[i] = np.array([point.x, point.y, point.z])
            shs[i] = (np.array([point.r, point.g, point.b]) / 255 - 0.5) / SH_C0_0
        rots = np.zeros([num_points, 4])
        rots[:, 0] = 1
        alphas = np.ones([num_points]) * 0.8
        pws = pws.astype(np.float32)
        rots = rots.astype(np.float32)
        alphas = alphas.astype(np.float32)
        shs = shs.astype(np.float32)

        N, D = pws.shape
        index = faiss.IndexFlatL2(D)
        index.add(pws)
        distances, indices = index.search(pws, 2)
        distances = np.clip(distances[:, 1], 0.01, 3)
        scales = distances[:, np.newaxis].repeat(3, 1)

        dtypes = [('pw', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('alpha', '<f4'),
                  ('sh', '<f4', (3,))]

        self.gs = np.rec.fromarrays([pws, rots, scales, alphas, shs], dtype=dtypes)

        twcs = torch.stack([x.twc for x in self.cameras])
        cam_dist = torch.linalg.norm(twcs - torch.mean(twcs, axis=0), axis=1)
        self.sence_size = float(torch.max(cam_dist)) * 1.1

    def __getitem__(self, index: int):
        return self.cameras[index], self.images[index]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]
