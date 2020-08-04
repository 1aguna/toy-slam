import numpy as np
import os
from pathlib import Path


class DataLoader:

    def __init__(self, data_path, idx=0, show_img=False, color=False):
        """
        :param data_path:  String representation to the path to the root of KITTI directory
        :param idx:         Optional index to start reading in the data
        :param color        Flag to show images. False by default
        :param color        Flag to show colored images. False by default
        """
        self.data_path = Path(data_path)
        self.lidar_path = self.data_path
        print(self.lidar_path)
        self.lidar_bin_path = self.lidar_path / "data"

        self.idx = idx
        self.color = color

        if self.color:
            self.img_path = self.data_path / "image_02/data"
        else:
            self.img_path = self.data_path / "/image_00/data"

        self.size = 0
        for _ in self.lidar_bin_path.iterdir():
            self.size += 1

    def loadPCL(self) -> np.array:
        bin_name = str(self.idx).zfill(10) + ".bin"
        current_bin = self.lidar_bin_path / bin_name
        self.idx += 1
        lidar_bin = np.fromfile(current_bin, dtype=np.float32)
        lidar_bin = lidar_bin.reshape((-1, 4))
        pcl_xyz = lidar_bin[:, :-1]
        return pcl_xyz

    def __repr__(self):
        # return "a={0},b={1}".format(a, b)
        return "Data Path: , Current Index: ".format(self.data_path, self.idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.size:
            return self.loadPCL()
        else:
            raise StopIteration
