import numexpr as ne
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.pardir)

from Project_2_dataExtraction import *
from cvxopt import matrix
from cvxopt import solvers
import time
import numpy as np
from sklearn.metrics import confusion_matrix

seed = 1679838
np.random.seed(seed)


def rbf_kernel(X, z, gamma):
    assert gamma > 0., 'gamma for RBF kernel must be > 0'
    X_norm = np.sum(X ** 2, axis=-1)
    y_norm = np.sum(z ** 2, axis=-1)
    return ne.evaluate('exp( -gamma * (A + B - 2 * C))', {
        'A': X_norm[:, None],
        'B': y_norm[None, :],
        'C': np.dot(X, z.T),
        'gamma': gamma
    })


def polynomial_kernel(X, z, gamma):
    assert gamma >= 1, 'gamma for polynomial kernel must be >= 1'
    return (X.dot(z.T) + 1) ** gamma


full_data = np.vstack((xLabel3, xLabel8))
data_norm = full_data / 255  # (2000x784)

full_labels = np.hstack((-np.ones((1000,)), np.ones((1000,))))  # (2000,)
#print(np.unique(full_labels, return_counts=True))
# a = np.ones((2000,))
# b = 0.5
N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)

#print('TRAIN ', np.unique(y_train, return_counts=True))
#print('TEST ', np.unique(y_test, return_counts=True))


class SVM():
    def __init__(self, kernel=polynomial_kernel, gamma=1, C=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma


    def predict(self, X, a=None, sv=None, sv_y=None, b=None):
        k = self.kernel
        if a is not None:
            return np.sign((a[None, :] * sv_y[None, :]).dot(k(X, sv, self.gamma).T) + b)
        else:
            return np.sign((self.a[None, :] * self.sv_y[None, :]).dot(k(X, self.sv, self.gamma).T) + self.b)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

    def _accuracy(self, X, y, a, sv, sv_y, b):
        y_pred = self.predict(X, a, sv, sv_y, b)
        return np.sum(y == y_pred) / len(y)
    
    def conf_mat(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y, y_pred)
        return cm

    def linesearch(self, X, y, tol=1e-6, q_value=2, num_iter=10):

        nfev = 0
        y = y  # .reshape(1600,1)

        n_samples, n_features = X.shape
        alpha_k = np.zeros(y.shape)

        alpha_k1 = np.zeros(y.shape)

        K = self.kernel(X, X, self.gamma)
        K = np.array(K)
        t = time.time()

        grad_q = -np.ones_like(y)
        L_neg = np.where((alpha_k < tol) & (y == -1))[0]
        L_pos = np.where((alpha_k < tol) & (y == 1))[0]
        U_neg = np.where((alpha_k >= self.C - tol) & (y == -1))[0]
        U_pos = np.where((alpha_k >= self.C - tol) & (y == 1))[0]
        SV = np.where((alpha_k > tol) & (alpha_k < self.C))[0]
        S = np.array(L_neg.tolist() + U_pos.tolist() + SV.tolist())
        R = np.array(L_pos.tolist() + U_neg.tolist() + SV.tolist())
        m_alpha = max((-grad_q * y)[R,])
        M_alpha = min((-grad_q * y)[S,])
        diff = m_alpha - M_alpha
        idx = 0
        while (diff > tol) and idx < num_iter:
            R_index = R[np.argsort((-grad_q * y)[R,], axis=0)[::-1][:q_value // 2]].flatten().tolist()
            S_index = S[np.argsort((-grad_q * y)[S,], axis=0)[:q_value // 2]].flatten().tolist()
            W_index = R_index + S_index

            Q = np.outer(y[W_index,], y[W_index,]) * K[W_index, :][:, W_index]
            Q2 = np.outer(y, y[W_index,]) * K[:, W_index]

            d_i = y[R_index]
            d_j = -y[S_index]
            d = np.hstack([d_i, d_j])
            t_star = - ((grad_q[W_index].T.dot(d)) / d.T.dot(Q).dot(d))
            # print('t_Star', t_star)

            alpha_k1[W_index,] = alpha_k[W_index,] + t_star * d
            alpha_k1[alpha_k1 < tol] = 0
            #grad_q[W_index,] += np.array(Q).dot(alpha_k1[W_index,] - alpha_k[W_index,])  # updating gradient
            grad_q += np.sum(np.array(Q2) * (alpha_k1[W_index,]-alpha_k[W_index,]), axis=1)
            alpha_k[W_index,] = alpha_k1[W_index,].copy()
            L_neg = np.where((alpha_k < tol) & (y == -1))[0]
            L_pos = np.where((alpha_k < tol) & (y == 1))[0]
            U_neg = np.where((alpha_k >= self.C - tol) & (y == -1))[0]
            U_pos = np.where((alpha_k >= self.C - tol) & (y == 1))[0]
            SV = np.where((alpha_k > tol) & (alpha_k < self.C))[0]
            S = np.array(list(set(L_neg.tolist() + U_pos.tolist() + SV.tolist())))
            # print(W_index)
            R = np.array(list(set(L_pos.tolist() + U_neg.tolist() + SV.tolist())))
            # assert len(R) == len(set(R)), 'doubles'
            m_alpha = max((-grad_q * y)[R,])
            M_alpha = min((-grad_q * y)[S,])
            diff = m_alpha - M_alpha
            idx += 1
           
        #print(f"\ndiff > tol: {c1} \ny.T.dot(alpha_k) < tol: {c2} \nidx < num_iter: {c3}")
        self.diff = diff
        self.iterations = idx
        self.fit_time = time.time() - t
        alpha_k = alpha_k.flatten()
        sv = alpha_k > 0
        self.sv = X[sv, :]
        self.sv_y = y[sv]
        self.a = alpha_k[sv]
        # print('iter', idx)

        self.b = np.mean(self.sv_y - (self.a * self.sv_y).dot(K[sv, :][:, sv]))



