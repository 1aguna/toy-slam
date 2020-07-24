import numpy as np
import os


class DataManager:

    def __init__(self, main_dir, lidar_dir, img_dir):
        self.main_dir = main_dir
        self.lidar_dir = lidar_dir
        self.img_Dir = img_dir

    def readBin(self, bin_path) -> np.array:
        lidar_bin = np.fromfile(bin_path, dtype=np.float32)
        lidar_bin = lidar_bin.reshape((-1, 4))
        pcl_xyz = lidar_bin[:, :-1]
        return pcl_xyz

    def __repr__(self):
        return "Data path: " + str(self.main_dir)