import copy

import numpy as np


class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance


class KNNResultSet:
    def __init__(self, nn_size):
        self.nn_size = nn_size
        self.count = 0
        self.worst_dis = float("inf")
        self.worst_dis_list = []
        for i in range(nn_size):
            self.worst_dis_list.append(DistIndex(self.worst_dis, 0))

    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.worst_dis_list):
            output += 'index : %d --- distance : %.2f\n' % (dist_index.index, dist_index.distance)
        return output

    def worst_distance(self):
        return self.worst_dis

    def add_point(self, dist, index):
        if dist > self.worst_dis:
            return
        if self.count < self.nn_size:
            self.count += 1
        i = self.count - 1
        while i > 0:
            if self.worst_dis_list[i - 1].distance > dist:
                self.worst_dis_list[i] = copy.deepcopy(self.worst_dis_list[i - 1])
                i -= 1
            else:
                break
        self.worst_dis_list[i].distance = dist
        self.worst_dis_list[i].index = index
        self.worst_dis = self.worst_dis_list[self.nn_size - 1].distance


class RadiusNNResultSet:
    def __init__(self, radius):
        self.count = 0
        self.radius = radius
        self.worst_dis_list = []

    def __str__(self):
        self.worst_dis_list.sort()
        output = ''
        for i, dist_index in enumerate(self.worst_dis_list):
            output += 'index : %d --- distance : %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\n' \
                  % (self.count, self.radius)
        return output

    def worst_radius(self):
        return self.radius

    def add_point(self, dist, index):
        if dist > self.radius:
            return
        self.count += 1
        self.worst_dis_list.append(DistIndex(dist, index))
