import numpy as np
from random import choice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''Input'''
'''This input is from ModelNet40/Car'''
# file = open('data/car/car_0107.txt')
# val_list = file.readlines()
# lists = []
#
# for string in val_list:
#     string = string.split(',', 6)
#     lists.append(string[0:3])
# X = np.array(lists)
# X = X.astype(float)
# print(X)
# x = X[:, 0]
# y = X[:, 1]
# z = X[:, 2]
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, z)
# plt.show()
'''This input is from ModelNet40/Car'''

'''This is the test input'''
X = np.random.randn(100, 3)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.xlim((-3, 3))
plt.ylim((-3, 3))
ax.set_zlim3d(-3, 3)
plt.show()
'''This is the test input'''

'''Determine the number of down sampled points'''
size_down = 10

'''Choose a point randomly'''
point_choice = np.zeros((size_down, 3))
index_X = choice(range(len(X)))
point_choice[0, :] = X[index_X, :]
X = np.delete(X, index_X, 0)

'''Down sample'''
dis_nearest = float("inf")
dis_farthest = 0
index_choose = 0
point_temp = np.zeros(1)

# TODO: How can we acc this process??
#       Binary Search Tree/ KD-Tree / Oc- Tree?
for index in range(size_down-1):
    for i in range(X.shape[0]):
        point_temp = X[i, :]
        for j in range(size_down):
            if not (point_choice[j, :] == [0, 0, 0]).all():
                temp = np.linalg.norm(x=(point_temp-point_choice[j, :]))
                if temp < dis_nearest:
                    dis_nearest = temp
            else:
                break
        '''We compute the distance to the Nearest FPS point
            and need to choose a Largest value'''
        if dis_nearest > dis_farthest:
            dis_farthest = dis_nearest
            index_choose = i
            dis_nearest = float("inf")
        else:
            dis_nearest = float("inf")
            continue
    point_choice[index+1, :] = X[index_choose, :]
    '''The chosen point cannot be choose again'''
    X = np.delete(X, index_choose, 0)
    dis_farthest = 0

print(point_choice)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(point_choice[:, 0], point_choice[:, 1], point_choice[:, 2])
plt.xlim((-3, 3))
plt.ylim((-3, 3))
ax.set_zlim3d(-3, 3)
plt.show()
