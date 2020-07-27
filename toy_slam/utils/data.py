import numpy as np
import os
from pathlib import Path


class DataManager:

    def __init__(self, data_path, idx=0, show_img=False, color=False):
        """
        :param data_path:  String representation to the path to the root of KITTI directory
        :param idx:         Optional index to start reading in the data
        :param color        Flag to show images. False by default
        :param color        Flag to show colored images. False by default
        """
        self.data_path = Path(data_path)
        self.lidar_path = self.data_path / "velodyne_points"
        self.lidar_bin_path = self.lidar_path / "data"

        self.idx = idx
        self.color = color

        if self.color:
            self.img_path = self.data_path / "image_02/data"
        else:
            self.img_path = self.data_path / "/image_00/data"

    def loadPCL(self) -> np.array:
        lidar_bin = np.fromfile(self.lidar_bin_path, dtype=np.float32)
        lidar_bin = lidar_bin.reshape((-1, 4))
        pcl_xyz = lidar_bin[:, :-1]
        return pcl_xyz

    def __repr__(self):
        # return "a={0},b={1}".format(a, b)
        return "Data Path: , Current Index: ".format(self.data_path, self.idx)
