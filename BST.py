import numpy as np
import time
from NN_container import KNNResultSet, RadiusNNResultSet
from NN_search import knn_search, radius_search
from tree_node import Node


time_start = time.time()
'''Input'''
db_size = 100
data = np.random.permutation(db_size).tolist()
print(data)
time_end = time.time()
# print('totally cost', time_end - time_start)
'''BST generate'''


def insert(root, key, value=-1):
    if root is None:
        root = Node(key, value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            pass
    return root


time_start = time.time()
root = None
for i, point in enumerate(data):
    # print("This is {}th point".format(i))
    root = insert(root, point, i)
time_end = time.time()
# print('totally cost', time_end - time_start)
'''BST generate'''
'''BST Search'''


def search_bst_recur(root, key):
    if root is None or root.key == key:
        return root
    else:
        if key < root.key:
            return search_bst_recur(root.left, key)
        else:
            return search_bst_recur(root.right, key)


def search_bst_iter(root, key):
    current_root = root
    while root is not None:
        if current_root.key == key:
            return current_root
        elif key < current_root.key:
            current_root = current_root.left
        else:
            current_root = current_root.right
    return current_root


'''BST Search'''
'''Test search'''
# time_start = time.time()
# root_s_recur = search_bst_recur(root, 7)
# time_end = time.time()
# print('totally cost', time_end - time_start)
# time_start = time.time()
# root_s_iter = search_bst_iter(root, 7)
# time_end = time.time()
# print('totally cost', time_end - time_start)
# print(root_s_recur.value)
# print(root_s_iter.value)
'''Test search'''
'''Inorder, Preorder,Postorder'''

data_order = []
def inorder(root):
    if root is not None:
        inorder(root.left)
        data_order.append(root.value)
        inorder(root.right)


def preorder(root):
    if root is not None:
        print(root.value)
        preorder(root.left)
        preorder(root.right)


def postorder(root):
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root.value)


'''Inorder'''
'''Test order'''
# inorder(root)
# print(data_order)
# data = np.array(data)[data_order]
# print(data)
# preorder(root)
# postorder(root)
'''Test order'''

'''NN search'''
# KNN
k = 5
query_key = 3
result_set_knn = KNNResultSet(5)
knn_search(root, result_set_knn, query_key)
print(result_set_knn)
for i in range(k):
    print(data[result_set_knn.worst_dis_list[i].index], end="/")


'''NN search'''


