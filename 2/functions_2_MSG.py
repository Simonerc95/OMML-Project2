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


def polynomial_kernel(X, z, gamma):
    assert gamma >= 1, 'gamma for polynomial kernel must be >= 1'
    return (X.dot(z.T) + 1) ** gamma


full_data = np.vstack((xLabel3, xLabel8))
data_norm = full_data / 255  # (2000x784)

full_labels = np.hstack((-np.ones((1000,)), np.ones((1000,))))  # (2000,)

N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)


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

    def conf_mat(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y, y_pred)
        return cm

    def decomp_method(self, X, y, tol=1e-6, q_value=10, num_iter=10):
        nfev = 0
        alpha_k = np.zeros(y.shape)
        alpha_k1 = np.zeros(y.shape)
        t = time.time()

        grad_q = -np.ones_like(y)
        L_neg = np.where((alpha_k < tol) & (y == -1))[0]
        L_pos = np.where((alpha_k < tol) & (y == 1))[0]
        U_neg = np.where((alpha_k >= self.C - tol) & (y == -1))[0]
        U_pos = np.where((alpha_k >= self.C - tol) & (y == 1))[0]
        SV = np.where((alpha_k > tol) & (alpha_k < self.C))[0]
        S = np.array(L_neg.tolist() + U_pos.tolist() + SV.tolist())
        R = np.array(L_pos.tolist() + U_neg.tolist() + SV.tolist())
        m_alpha = max((-grad_q * y)[R])
        M_alpha = min((-grad_q * y)[S])
        diff = m_alpha - M_alpha
        idx = 0

        while (diff > 1e-2) and idx < num_iter:
            R_index = R[np.argsort((-grad_q * y)[R,], axis=0)[::-1][:q_value // 2]].flatten().tolist()
            S_index = S[np.argsort((-grad_q * y)[S,], axis=0)[:q_value // 2]].flatten().tolist()
            W_index = R_index + S_index

            # Building matrices for cvxopt

            P = matrix(np.outer(y[W_index], y[W_index]) * self.kernel(X[W_index, :], X[W_index, :], self.gamma),
                       tc='d')
            Q = np.outer(y, y[W_index]) * self.kernel(X, X[W_index, :], self.gamma)
            q = matrix(np.delete(Q, W_index, axis=0).T.dot(np.delete(alpha_k, W_index, axis=0)) - \
                       np.ones(q_value), (q_value, 1), tc='d')
            A = matrix(y[W_index].dot(np.identity(q_value)), (1, q_value), tc='d')
            b = matrix(-np.delete(y, W_index, axis=0).T.dot(np.delete(alpha_k, W_index, axis=0)), tc='d')
            G = matrix(np.vstack((-1 * np.identity(q_value), np.identity(q_value))), tc='d')
            h = matrix(np.vstack((np.zeros((q_value, 1)), self.C * np.ones((q_value, 1)))), tc='d')

            solvers.options['show_progress'] = False
            solution = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

            nfev += solution['iterations']

            alpha_k1[W_index,] = np.array(solution['x']).flatten()
            alpha_k1[alpha_k1 < tol] = 0

            grad_q += np.sum(Q * (alpha_k1[W_index,] - alpha_k[W_index,]), axis=1)

            alpha_k[W_index,] = alpha_k1[W_index,].copy()

            L_neg = np.where((alpha_k < tol) & (y == -1))[0]
            L_pos = np.where((alpha_k < tol) & (y == 1))[0]
            U_neg = np.where((alpha_k >= self.C - tol) & (y == -1))[0]
            U_pos = np.where((alpha_k >= self.C - tol) & (y == 1))[0]
            SV = np.where((alpha_k >= tol) & (alpha_k < self.C - tol))[0]
            S = np.array(list(set(L_neg.tolist() + U_pos.tolist() + SV.tolist())))
            R = np.array(list(set(L_pos.tolist() + U_neg.tolist() + SV.tolist())))
            m_alpha = max((-grad_q * y)[R])
            M_alpha = min((-grad_q * y)[S])
            diff = m_alpha - M_alpha
            idx += 1

            # Evaluation of objective function
            # if idx == 1 or diff <= 1e-2 or idx == num_iter:
            #     print(f'dual objective at step {idx}', solution['primal objective'])
            #     e = np.ones_like(alpha_k)
            #     print(f'dual objective at step {idx} with formula', 0.5 * alpha_k.T.dot(np.outer(y, y) * \
            #                                             self.kernel(X,X, self.gamma)).dot(alpha_k) - e.T.dot(alpha_k))

        self.diff = diff
        self.iterations = idx
        self.fit_time = time.time() - t
        self.nfev = nfev
        alpha_k = alpha_k.flatten()
        sv = alpha_k > 0
        self.sv = X[sv, :]
        self.sv_y = y[sv]
        self.a = alpha_k[sv]
        self.b = np.mean(self.sv_y - (self.a * self.sv_y).dot(self.kernel(X[sv], X[sv], self.gamma)))
