"""
ref: https://github.com/kissb2/PyICP-SLAM/blob/dc026afff934aa0867f1a67a5e5b01a145ce1109/utils/ScanContextManager.py
"""

import numpy as np
from scipy import spatial
import g2o


class ScanContext:
    def __init__(self, shape=[20, 60], num_candidates=10,
                 threshold=0.15):  # defualt configs are same as the original paper
        self.shape = shape
        self.num_candidates = num_candidates
        self.threshold = threshold

        self.max_length = 80  # recommended but other (e.g., 100m) is also ok.

        self.SIZE = 15000  # capable of up to SIZE number of nodes
        self.pcls = [None] * self.SIZE
        self.scan_contexts = [None] * self.SIZE
        self.ringkeys = [None] * self.SIZE

        self.curr_node_idx = 0

    def addNode(self, node_idx, pcl):
        sc = self.pcl2sc(pcl, self.shape, self.max_length)
        rk = self.scan2rk(sc)

        self.curr_node_idx = node_idx
        self.pcls[node_idx] = pcl
        self.scan_contexts[node_idx] = sc
        self.ringkeys[node_idx] = rk

    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        x = point[0]
        y = point[1]
        # z = point[2]

        if (x == 0.0):
            x = 0.001
        if (y == 0.0):
            y = 0.001

        theta = xy2theta(x, y)
        faraway = np.sqrt(x * x + y * y)

        idx_ring = np.divmod(faraway, gap_ring)[0]
        idx_sector = np.divmod(theta, gap_sector)[0]

        if (idx_ring >= num_ring):
            idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

        return int(idx_ring), int(idx_sector)

    def pcl2sc(self, pcl, sc_shape, max_length):
        num_ring = sc_shape[0]
        num_sector = sc_shape[1]

        gap_ring = max_length / num_ring
        gap_sector = 360 / num_sector

        enough_large = 500
        sc_storage = np.zeros([enough_large, num_ring, num_sector])
        sc_counter = np.zeros([num_ring, num_sector])

        num_points = pcl.shape[0]
        for pt_idx in range(num_points):
            point = pcl[pt_idx, :]
            point_height = point[2] + 2.0  # for setting ground is roughly zero

            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, num_ring, num_sector)

            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] = sc_counter[idx_ring, idx_sector] + 1

        sc = np.amax(sc_storage, axis=0)

        return sc

    def scan2rk(self, sc):
        return np.mean(sc, axis=1)

    def getPcl(self, node_idx):
        return self.pcls[node_idx]

    def scan_distance(self, scan1, scan2):
        num_sectors = scan1.shape[1]

        # repeate to move 1 columns
        _one_step = 1  # const
        sim_for_each_cols = np.zeros(num_sectors)
        for i in range(num_sectors):
            # Shift
            scan1 = np.roll(scan1, _one_step, axis=1)  # column shift

            # compare
            cos_similarity_sum = 0
            n_cols = 0
            for j in range(num_sectors):
                col_j_1 = scan1[:, j]
                col_j_2 = scan2[:, j]
                cos_similarity_sum = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))

                n_cols += 1

            # save
            sim_for_each_cols[i] = cos_similarity_sum / n_cols

        yaw_diff = np.argmax(sim_for_each_cols) + 1  # because python starts with 0
        sim = np.max(sim_for_each_cols)
        dist = 1 - sim

        return dist, yaw_diff

    def detectLoop(self):
        exclude_recent_nodes = 30
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes

        if valid_recent_node_idx < 1:
            return None, None, None
        else:
            # step 1
            ringkey_history = np.array(self.ringkeys[:valid_recent_node_idx])
            ringkey_tree = spatial.KDTree(ringkey_history)

            ringkey_query = self.ringkeys[self.curr_node_idx]
            _, nncandidates_idx = ringkey_tree.query(ringkey_query, k=self.num_candidates)

            # step 2
            query_sc = self.scan_contexts[self.curr_node_idx]

            nn_dist = 1.0  # initialize with the largest value of distance
            nn_idx = None
            nn_yawdiff = None

            for ith in range(self.num_candidates):
                candidate_idx = nncandidates_idx[ith]
                candidate_sc = self.scan_contexts[candidate_idx]
                dist, yaw_diff = self.scan_distance(candidate_sc, query_sc)
                if dist < nn_dist:
                    nn_dist = dist
                    nn_yawdiff = yaw_diff
                    nn_idx = candidate_idx

            if nn_dist < self.threshold:
                nn_yawdiff_deg = nn_yawdiff * (360 / self.shape[1])
                return nn_idx, nn_dist, nn_yawdiff_deg  # loop detected!
            else:
                return None, None, None
