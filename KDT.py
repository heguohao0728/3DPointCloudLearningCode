import numpy as np
import math
import time
from KD_tree_node import Node
from NN_container import KNNResultSet, RadiusNNResultSet


def kdtree_recur_bulid(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)
    if len(point_indices) > leaf_size:
        point_indices_sorted, _ = sort_key_by_value(point_indices, db[point_indices, axis])
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) / 2.0
        root.left = kdtree_recur_bulid(root.left, db, point_indices_sorted[0: middle_right_idx],
                                       axis_round_robin(axis, dim=db.shape[1]), leaf_size)
        root.right = kdtree_recur_bulid(root.right, db, point_indices_sorted[middle_right_idx:],
                                        axis_round_robin(axis, dim=db.shape[1]), leaf_size)
    return root


# search x-y-z-x-y-z-x.....
def axis_round_robin(axis, dim):
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


def sort_key_by_value(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]
    root = None
    root = kdtree_recur_bulid(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kd_knn(root : Node, db, result_set : KNNResultSet, query):
    if root is None:
        return False
    if root.is_leaf():
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return True
    if query[root.axis] <= root.value:
        kd_knn(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kd_knn(root.right, db, result_set, query)
    else:
        kd_knn(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_distance():
            kd_knn(root.left, db, result_set, query)
    return True


def radius_search(root : Node, db, result_set : RadiusNNResultSet, query):
    if root is None:
        return False
    if root.is_leaf():
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return True
    if query[root.axis] < root.value:
        radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_radius():
            radius_search(root.right, db, result_set, query)
    else:
        radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worst_radius():
            radius_search(root.left, db, result_set, query)
    return True


if __name__ == '__main__':
    db_size = 200
    dim = 3
    leaf_size = 4
    k = 2
    db_np = np.random.rand(db_size, dim)
    root = kdtree_construction(db_np, leaf_size=leaf_size)

    query = np.asarray([0.1, 0.1, 0.1])

    # Normal Search
    time_start = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])
    print(" ")
    time_end = time.time()
    print('totally cost', time_end - time_start)
    print("=====================================")

    # Knn Search
    time_start = time.time()
    result_set = KNNResultSet(k)
    kd_knn(root, db_np, result_set, query)
    print(result_set)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    # for i in range(k):
    #     print(db_np[result_set.worst_dis_list[i].index])

    # Radius Search
    result_set = RadiusNNResultSet(radius=0.2)
    radius_search(root, db_np, result_set, query)
    print(result_set)


