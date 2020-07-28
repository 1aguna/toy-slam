import numpy as np
import random
import spatialmath as sm


def down_sample(original_arr, n_pts):
    """
    :param original_arr:    np.array of original points
    :param n_pts:           Size of downsampled array
    :return:                np.array of
    """

    assert original_arr.shape[0] > n_pts
    idxs = np.random.choice(original_arr.shape[0], n_pts, replace=False)
    down_sampled_pts = original_arr[idxs, :]

    return down_sampled_pts


def xy2theta(x, y):
    if x >= 0 and y >= 0:
        theta = 180 / np.pi * np.arctan(y / x);
    if x < 0 and y >= 0:
        theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)))
    if x < 0 and y < 0:
        theta = 180 + ((180 / np.pi) * np.arctan(y / x))
    if x >= 0 and y < 0:
        theta = 360 - ((180 / np.pi) * np.arctan((-y) / x))

    return theta
