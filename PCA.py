import numpy as np

'''This code generate a principle Vector(Matrix) of Matrix X'''

'''input'''
X = np.random.randn(10, 10)
print(X)
'''input'''

'''Normalize'''
X_sum = np.zeros((X.shape[0]))
for i in range(X.shape[1]):
    X_sum = X_sum + X[:, i]
avg = X_sum/X.shape[1]
avg = np.reshape(avg, (-1, 1))
X = X - avg
'''Normalize'''

'''SVD'''
U, s, V = np.linalg.svd(X)
'''SVD'''

'''Principle Vector'''
down_dim = 3
Z = np.ndarray((X.shape[0], down_dim))
for i in range(down_dim):
    Z[:, i] = U[:, i]
print(Z)
'''Principle Vector'''



