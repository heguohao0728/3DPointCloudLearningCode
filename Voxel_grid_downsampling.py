import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import choice, sample

'''input'''
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
X = np.random.randn(20, 3)
print(X)
x_total_num = X.shape[0]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.xlim((-3, 3))
plt.ylim((-3, 3))
ax.set_zlim3d(-3, 3)
plt.show()

'''Compute min and max'''
max_index = np.argmax(X, 0)
x_max = X[max_index[0], 0]
y_max = X[max_index[1], 1]
z_max = X[max_index[2], 2]
min_index = np.argmin(X, 0)
x_min = X[min_index[0], 0]
y_min = X[min_index[1], 1]
z_min = X[min_index[2], 2]

'''Determine voxel size(by yourself)'''
size_r = 2

'''Compute the dim of voxel grid'''
D_x = (x_max - x_min) / size_r
D_y = (y_max - y_min) / size_r
D_z = (z_max - z_min) / size_r

'''Compute the voxel index of each point'''
h = np.zeros(x_total_num)
for i in range(x_total_num):
    h_x = (X[i, 0] - x_min) // size_r
    h_y = (X[i, 1] - y_min) // size_r
    h_z = (X[i, 2] - z_min) // size_r
    h[i] = h_x + h_y * D_x + h_z * D_x * D_y
point_index = np.argsort(h)

'''Down_sampling'''
temp = h[point_index[0]]
point_temp = []
point_choice = []
kind = 1
for i in range(x_total_num):
    if temp == h[point_index[i]]:
        point_temp.append(X[point_index[i]])
        continue
    else:
        kind = kind + 1
        temp = h[point_index[i]]
        '''Choose random'''
        # point_choice.append(choice(point_temp))
        '''Choose average'''
        # point_choice_arr = np.array(point_temp)
        # point_choice_num = point_choice_arr.shape[0]
        # point_choice.append(np.sum(point_choice_arr, axis=0)/point_choice_num)
        '''Choose mid'''
        point_choice_arr = np.array(point_temp)
        point_choice_num = point_choice_arr.shape[0]
        point_choice.append(point_choice_arr[(point_choice_num//2), :])

        point_temp.clear()
        point_temp.append(X[point_index[i]])
print("We can split {} class of point(Down sample)".format(kind))

point_choice = np.array(point_choice)
print(point_choice)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(point_choice[:, 0], point_choice[:, 1], point_choice[:, 2])
plt.xlim((-3, 3))
plt.ylim((-3, 3))
ax.set_zlim3d(-3, 3)
plt.show()
