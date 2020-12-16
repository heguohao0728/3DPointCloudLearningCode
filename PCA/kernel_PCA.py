import numpy as np
import math
'''Input'''
X = np.random.randn(10, 12)
print(X)
x_total_num = X.shape[0]

'''Select The Kernel Matrix'''
'''k=(1+xi*xj)^2'''
'''k=e^(1+xi*xj)'''
'''........'''
'''Choose with experiment'''
K = np.zeros((x_total_num, x_total_num))
for i in range(x_total_num):
    for j in range(x_total_num):
        K[i, j] = (1+np.matmul(X[i, :], X[j, :].T))**2

'''Normalize K'''
I_n = np.ones((x_total_num, x_total_num))/x_total_num
K_n = K - 2*np.matmul(I_n, K) + np.matmul(np.matmul(I_n, K), I_n)

'''Solve the eigenvector'''
e_value, e_vector = np.linalg.eig(K_n)

'''Normalize eigenvector'''
norm_e_vector = np.zeros((e_vector.shape[0], e_vector.shape[1]))
for i in range(x_total_num):
    norm_e_vector[:, i] = e_vector[:, i]/np.sqrt(e_value[i])
index = np.argsort(e_value)
print(index)
'''Projection Matrix'''
down_dim = 5
Z = np.zeros((e_vector.shape[0],down_dim))
j = index.shape[0]-1
for i in range(down_dim):
    Z[:, i] = norm_e_vector[:, index[j]]
    j = j-1
print(Z)



