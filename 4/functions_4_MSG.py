import numexpr as ne
from Project_2_dataExtraction import *
from cvxopt import matrix
from cvxopt import solvers
import time
import numpy as np
import pandas as pd

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
data_norm = full_data / 255  # (2000x784)

full_labels = np.hstack((np.full((1000,), 3), np.full((1000,), 6), np.full((1000,), 8)))  # (2000,)

N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)

# print("Y_train: ", np.unique(y_train, return_counts=True))
# print("Y_test: ", np.unique(y_test, return_counts=True))
# print('TRAIN ', np.unique(y_train, return_counts=True))
# print('TEST ',np.unique(y_test, return_counts=True))


class SVM():
    def __init__(self, kernel=polynomial_kernel, gamma=1, C=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.a = {}
        self.sv = {}
        self.sv_y = {}
        self.opt_sol = {}
        self.b = {}

    def fit(self, X, y, tol=1e-7, label=None):
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

        # Support vectors have non zero lagrange multipliers
        sv = a > tol
        ind = np.arange(len(a))[sv]
        if label is None:
            self.a = a[sv]
            self.sv = X[sv]
            self.sv_y = y[sv]
            # print("%d support vectors out of %d points" % (len(self.a), n_samples))
            self.opt_sol = solution
            # Intercept
            self.b = 0
            for n in range(len(self.a)):
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
            self.b /= len(self.a)
        else:            
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

        # Weight vector

    def predict(self, X, a=None, sv=None, sv_y=None, b=None):
        k = self.kernel
        if a is not None:
            return np.sign((a[None, :] * sv_y[None, :]).dot(k(X, sv, self.gamma).T) + b)
        else:
            return np.sign((self.a[None, :] * self.sv_y[None, :]).dot(k(X, self.sv, self.gamma).T) + self.b)
        
    def predict_df(self, X, lab, a=None, sv=None, sv_y=None, b=None):
        k = self.kernel
        if a is not None:
            return (a[None, :] * sv_y[None, :]).dot(k(X, sv, self.gamma).T) + b
        else:
            return (self.a[lab][None, :] * self.sv_y[lab][None, :]).dot(k(X, self.sv[lab], self.gamma).T) + self.b[lab]

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

    def _accuracy(self, X, y, a, sv, sv_y, b):
        y_pred = self.predict(X, a, sv, sv_y, b)
        return np.sum(y == y_pred) / len(y)

    def test_loss(self, X, y):
        pass

    def output(self):
        print(self.opt_sol)
        
    def fit_multi(self, X_t, y_t):
        self.labels = np.unique(y_t)
        for lab in self.labels:
            y_temp = y_t.copy()
            y_temp[y_t == lab] = 1
            y_temp[y_t != lab] = -1
            self.fit(X_t, y_temp, label=lab)
            
    def predict_multi(self, X_t):
        out = pd.DataFrame([], columns=self.labels)
        for lab in self.labels:
            out.loc[:, lab] = self.predict_df(X_t, lab)[0]
        return out#.idxmax(axis=1).tolist()
            
    def accuracy_multi(self, X_t, y_t):
        y_pred = self.predict_multi(X_t).idxmax(axis=1).to_numpy()
        return np.sum(y_pred == y_t)/len(y_t)

clf = SVM(C=10,gamma=2)
# clf.decomp_method(X_train, y_train, q_value=20, num_iter=1000)
clf.fit_multi(X_train, y_train)
ser = clf.predict_multi(X_train)
print(clf.accuracy_multi(X_test, y_test))
# prova.idxmax(axis=1).tolist()
# prova.iloc[0,:]
# print(clf.accuracy(X_test, y_test))
#%%
# clf = SVM(C=2,gamma=2)
# clf.decomp_method(X_train, y_train, q_value=800, num_iter=1000)
# print(clf.accuracy(X_test, y_test))

# # len(X_train[clf.a.flatten() >0, :])
# # clf.a#[None, :]
# # len(clf.a * clf.sv_y)

# clf2 = SVM(C=2,gamma=2)
# clf2.fit(X_train, y_train)
# print(clf2.accuracy(X_test, y_test))

#
# from joblib import Parallel, delayed
#
# def get_svm(c, g, q, i, ker):
#     clf = SVM(C=c,gamma=g, kernel=ker)
#     clf.decomp_method(X_train, y_train, q_value=q, num_iter=i)
#     return clf.accuracy(X_test, y_test)
#
# r = range(2,100,2)
# out = {}
# gam = (2, 10,)
# j = 0
#
# import matplotlib.pyplot as plt
# for  k in (polynomial_kernel,): #, rbf_kernel,):
#     out[k.__name__] = Parallel(n_jobs=-1, verbose=10)\
#         (delayed(get_svm)(c=2, g=gam[j], q=x, i=1000, ker=k) for x in r)
#     j += 1
#     plt.plot(list(r),out[k.__name__])
#     plt.title(f'{k.__name__.title()} Kernel')
#     plt.ylim(0.6, 1)
#     plt.show()
#
# #%%
# def moving_average(x, k):
#     return np.convolve(x, np.ones(k), 'valid') / k
#
# k = 21
# plt.plot(list(r[int(k/2):-int(k/2)]), moving_average(out, k))
# plt.show()
# #%%
# def get_svm2(c, g, q, i, tol):
#     clf = SVM(C=c,gamma=g)
#     clf.decomp_method(X_train, y_train, q_value=q, num_iter=i)
#     return clf.accuracy(X_test, y_test)
#
# r = range(2,800,10)
# out = Parallel(n_jobs=-1, verbose=10)\
#         (delayed(get_svm2)(c=2, g=2, q=2, i=1000, tol=x) for x in r)
