import numpy as np
from BST_tree_node import Node
from NN_container import KNNResultSet, RadiusNNResultSet
import math


def knn_search(root : Node, result_set : KNNResultSet, key):
    if root is None:
        return False
    result_set.add_point(math.fabs(root.key-key), root.value)
    if result_set.worst_distance() == 0:
        return True
    if key <= root.key:
        if knn_search(root.left, result_set, key):
            return True
        elif math.fabs(root.key - key) < result_set.worst_distance():
            return knn_search(root.right, result_set, key)
        return False
    else:
        if knn_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key - key) < result_set.worst_distance():
            return knn_search(root.left, result_set, key)
        return False


def radius_search(root : Node, result_set : RadiusNNResultSet, key):
    if root is None:
        return False
    result_set.add_point(math.fabs(root.key-key), root.value)
    if key <= root.key:
        if radius_search(root.left, result_set, key):
            return True
        elif math.fabs(root.key - key) < result_set.worst_radius():
            return radius_search(root.right, result_set, key)
        return False
    else:
        if radius_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key - key) < result_set.worst_radius():
            return radius_search(root.left, result_set, key)
        return False
