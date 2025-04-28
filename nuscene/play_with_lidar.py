import matplotlib.pyplot as plt
from .dataset import NuSceneData, CAM_LABELS

data = NuSceneData()

for label in CAM_LABELS:
    points, mask, depths = data.lidar_data.to_img_coord(data.cam_sensors[label])
    points = points[:, mask]
    coloring = depths[mask]
    plt.imshow(data.cam_sensors[label].data)
    plt.scatter(points[0, :], points[1, :], c=coloring, s=5)
    plt.show()
