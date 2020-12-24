import numpy as np

alpha = np.array([0,0])
alpha_1 = np.array([1,1])
q = np.array([[1,2],[2,1]])
p = q.dot(alpha_1-alpha)

l = np.array([[ 0,  3],
       [12, 15]])
print(l.shape)