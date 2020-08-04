import os
import argparse
import time
from toy_slam import DataLoader as dl
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="../data/2011_09_29/2011_09_29_drive_0071_sync/")
parser.add_argument("--maxIters", type=str, default=20)
parser.add_argument("--loop_threshold", type=float,
                    default=0.11)  # 0.11 is usually safe (for avoiding false loop closure)
parser.add_argument("--color", type=bool, default=True)
args = parser.parse_args()

lidar_dir = os.path.join(args.dir, "velodyne_points")
if args.color:  # color images
    img_dir = os.path.join(args.dir, "image_02", "data")
else:  # black and white images
    img_dir = os.path.join(args.dir, "image_00", "data")

dataset = dl.DataLoader(lidar_dir, show_img=True, color=False)

start = time.time()
for pcl in tqdm(dataset):
    pass
    # print(pcl.shape)
end = time.time()

elapsed = str(end - start)
print("Loaded %d pointclouds in %s seconds" % (dataset.size, elapsed))
