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
import pandas as pd
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


full_data = np.vstack((xLabel3, xLabel6, xLabel8))
data_norm = full_data / 255  # (3000x784)

full_labels = np.hstack((np.full((1000,), 3), np.full((1000,), 6), np.full((1000,), 8)))  # (2000,)

N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)


class SVMMulticlass():
    def __init__(self, kernel=polynomial_kernel, gamma=1, C=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.a = {}
        self.sv = {}
        self.sv_y = {}
        self.opt_sol = {}
        self.b = {}
        self.diff = {}
        self.iter = 0

    def fit(self, X, y, tol=1e-6, label=None):
        assert len(np.unique(y)) == 2, "The labels must be 2. For multiclass problems please run fit multi"
        n_samples, n_features = X.shape

        # Gram matrix
        K = self.kernel(X, X, self.gamma)

        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_samples) * -1)
        A = matrix(y.dot(np.identity(len(y))), (1, n_samples))

        b = matrix(0.0)
        G = matrix(np.vstack((-1 * np.identity(n_samples), np.identity(n_samples))))
        h = matrix(np.vstack((np.zeros((n_samples, 1)), self.C * np.ones((n_samples, 1)))))


        # solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        a = np.array(solution['x']).flatten()
        self.iter += solution['iterations']
        # Support vectors have non zero lagrange multipliers
        sv = a > tol
        ind = np.arange(len(a))[sv]
        
        self.a[label] = a[sv]
        self.sv[label] = X[sv]
        self.sv_y[label] = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))
        self.opt_sol[label] = solution
        # Intercept
        self.b[label] = 0
        for n in range(len(self.a[label])):
            self.b[label] += self.sv_y[label][n]
            self.b[label] -= np.sum(self.a[label] * self.sv_y[label] * K[ind[n], sv])
        self.b[label] /= len(self.a[label])
        
        '''
        KKT Condition check
        '''

        grad_q = a.T.dot(P)-np.ones_like(y)
        L_neg = np.where((a < tol) & (y == -1))[0]
        L_pos = np.where((a < tol) & (y == 1))[0]
        U_neg = np.where((a >= self.C - tol) & (y == -1))[0]
        U_pos = np.where((a >= self.C - tol) & (y == 1))[0]
        SV = np.where((a > tol) & (a < self.C))[0]
        S = np.array(L_neg.tolist() + U_pos.tolist() + SV.tolist())
        R = np.array(L_pos.tolist() + U_neg.tolist() + SV.tolist())
        m_alpha = max((-grad_q * y)[R,])
        M_alpha = min((-grad_q * y)[S,])
        self.diff[label] =  m_alpha - M_alpha

        
    def predict_df(self, X, lab, a=None, sv=None, sv_y=None, b=None):
        k = self.kernel
        if a is not None:
            return (a[None, :] * sv_y[None, :]).dot(k(X, sv, self.gamma).T) + b
        else:
            return (self.a[lab][None, :] * self.sv_y[lab][None, :]).dot(k(X, self.sv[lab], self.gamma).T) + self.b[lab]

        
    def fit_multi(self, X_t, y_t):
        t = time.time()
        self.labels = np.unique(y_t)
        for lab in self.labels:
            y_temp = y_t.copy()
            y_temp[y_t == lab] = 1
            y_temp[y_t != lab] = -1
            self.fit(X_t, y_temp, label=lab)
        self.fit_time = time.time()-t
            
    def predict_multi(self, X_t):
        out = pd.DataFrame([], columns=self.labels)
        for lab in self.labels:
            out.loc[:, lab] = self.predict_df(X_t, lab)[0]
        return out.idxmax(axis=1).to_numpy()
    
    def conf_mat(self, X, y):
        y_pred = self.predict_multi(X)
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y, y_pred)
        return cm
            
    def accuracy_multi(self, X_t, y_t):
        y_pred = self.predict_multi(X_t)
        return np.sum(y_pred == y_t)/len(y_t)

