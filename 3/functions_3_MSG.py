import numexpr as ne
from Project_2_dataExtraction import *
from cvxopt import matrix
from cvxopt import solvers
import time
import numpy as np

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
print(np.unique(full_labels, return_counts=True))
# a = np.ones((2000,))
# b = 0.5
N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)

print('TRAIN ', np.unique(y_train, return_counts=True))
print('TEST ', np.unique(y_test, return_counts=True))


# y = y_train.reshape(-1,1) * 1.
#
# P = matrix(y.dot((poly_kernel(X_train, X_train, 2).dot(y) * 1.).T), tc='d')
# q = matrix(-np.ones((y_train.shape[0])), tc='d')
# G = matrix(np.vstack([np.identity(X_train.shape[0]), -np.identity(X_train.shape[0])]), tc='d')
# c = 10
# C = np.ones(X_train.shape[0]) * c
# h = matrix(np.hstack([C, np.zeros(X_train.shape[0]).T]), tc='d')
# A = matrix(y_train.reshape(1,-1), tc='d')
# b = matrix(np.zeros(1), tc='d')

#
#
# dic = solvers.qp(P, q, G, h, A, b)
# arr = np.array(dic['x'])
# print(len(arr[arr>1e-4]))


class SVM():
    def __init__(self, kernel=polynomial_kernel, gamma=1, C=None):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = self.kernel(X, X, self.gamma)

        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(n_samples) * -1)
        A = matrix(y, (1, n_samples))
        b = matrix(0.0)
        G = matrix(np.vstack((-1 * np.identity(n_samples), np.identity(n_samples))))
        h = matrix(np.vstack((np.zeros((n_samples, 1)), self.C * np.ones((n_samples, 1)))))

        # solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        self.opt_sol = solution

        a = np.array(solution['x']).flatten()

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector

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

    def test_loss(self, X, y):
        pass

    def output(self):
        print(self.opt_sol)

    def decomp_method(self, X, y, tol=1e-12, q_value=10, num_iter=10):
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
        # assert len(R) == len(set(R)), 'doubles'
        m_alpha = max((-grad_q * y)[R,])
        # print('m_alpha', m_alpha)
        M_alpha = min((-grad_q * y)[S,])
        # print('M_alpha', M_alpha)
        diff = m_alpha - M_alpha
        # print('diff', diff)
        idx = 0
        while (diff > tol and y.T.dot(alpha_k) < tol) and idx < num_iter:
            W_index = R[np.argsort((-grad_q * y)[R,], axis=0)[::-1][:q_value // 2]].flatten().tolist() + \
                      S[np.argsort((-grad_q * y)[S,], axis=0)[:q_value // 2]].flatten().tolist()

            # W_index = np.random.choice(R,5).tolist() + np.random.choice(S,5).tolist()
            # print('grad_S',np.argsort((-grad_q*y)[R,], axis=0)[:q_value//2])
            # print('w_index', W_index)
            # print(grad_q[W_index])
            # Gram matrix
            # print('outer', np.outer(y[W_index,], y[W_index,]).shape)
            # print('k', K[W_index, :][:, W_index].shape)

            P = matrix(np.outer(y[W_index,], y[W_index,]) * K[W_index, :][:, W_index])
            # Q = np.outer(y, y[W_index,]) * K[:, W_index]
            q = matrix(grad_q[W_index])  # matrix(np.ones(q_value) * -1)
            A = matrix(y[W_index,].dot(np.identity(len(W_index))), (1, q_value))
            b = matrix(0.0)
            G = matrix(np.vstack((-1 * np.identity(q_value), np.identity(q_value))))
            h = matrix(np.vstack((np.zeros((q_value, 1)), self.C * np.ones((q_value, 1)))))

            # solve QP sub-problem
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            # self.opt_sol = solution
            nfev += solution['iterations']

            alpha_k1[W_index,] = np.array(solution['x']).flatten()
            alpha_k1[alpha_k1 < tol] = 0
            # print('graddot', alpha_k1[W_index,])
            # print('grad-upt', alpha_k1[W_index]-alpha_k[W_index])
            grad_q[W_index,] += np.array(P).dot(alpha_k1[W_index,] - alpha_k[W_index,])  # updating gradient
            # grad_q += np.sum(np.array(Q) * (alpha_k1[W_index,]-alpha_k[W_index,]), axis=1)
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
            # print(diff)

            # =============================================================================
            #             current_loss = solution['dual objective']
            #
            #             if current_loss < best_loss:
            #                 patience = 0
            #                 self.a = alpha_upt[sv_ind]
            #                 self.sv = sv
            #                 self.sv_y = sv_y
            #                 # Intercept
            #                 self.b = b
            #                 best_loss = current_loss
            #
            #             else:
            #                 if patience == tol:
            #                     break
            # =============================================================================
            idx += 1
            c1 = diff > tol
            c2 = y.T.dot(alpha_k) < tol
            c3 = idx < num_iter

        print(f"\ndiff > tol: {c1} \ny.T.dot(alpha_k) < tol: {c2} \nidx < num_iter: {c3}")
        self.iterations = idx
        print(self.iterations)
        self.fit_time = time.time() - t
        self.nfev = nfev
        alpha_k = alpha_k.flatten()
        sv = alpha_k > 0
        self.sv = X[sv, :]
        self.sv_y = y[sv]
        self.a = alpha_k[sv]
        # print('iter', idx)

        self.b = np.mean(self.sv_y - (self.a * self.sv_y).dot(K[sv, :][:, sv]))
        # print('Second b: ', self.b)

    def linesearch(self, X, y, tol=1e-12, q_value=10, num_iter=10):

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
        while (diff > tol and y.T.dot(alpha_k) < tol) and idx < num_iter:
            R_index = R[np.argsort((-grad_q * y)[R,], axis=0)[::-1][:q_value // 2]].flatten().tolist()
            S_index = S[np.argsort((-grad_q * y)[S,], axis=0)[:q_value // 2]].flatten().tolist()
            W_index = R_index + S_index

            Q = np.outer(y[W_index,], y[W_index,]) * K[W_index, :][:, W_index]
            # Q2 = np.outer(y, y[W_index,]) * K[:, W_index]

            d_i = y[R_index]
            d_j = -y[S_index]
            d = np.hstack([d_i, d_j])
            t_star = - ((grad_q[W_index].T.dot(d)) / d.T.dot(Q).dot(d))
            # print('t_Star', t_star)

            alpha_k1[W_index,] = alpha_k[W_index,] + t_star * d
            alpha_k1[alpha_k1 < tol] = 0
            grad_q[W_index,] += np.array(Q).dot(alpha_k1[W_index,] - alpha_k[W_index,])  # updating gradient
            # grad_q += np.sum(np.array(Q2) * (alpha_k1[W_index,]-alpha_k[W_index,]), axis=1)
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
            # print(diff)

            idx += 1
            c1 = diff > tol
            c2 = y.T.dot(alpha_k) < tol
            c3 = idx < num_iter

        print(f"\ndiff > tol: {c1} \ny.T.dot(alpha_k) < tol: {c2} \nidx < num_iter: {c3}")
        self.iterations = idx
        print(self.iterations)
        self.fit_time = time.time() - t
        self.nfev = nfev
        alpha_k = alpha_k.flatten()
        sv = alpha_k > 0
        self.sv = X[sv, :]
        self.sv_y = y[sv]
        self.a = alpha_k[sv]
        # print('iter', idx)

        self.b = np.mean(self.sv_y - (self.a * self.sv_y).dot(K[sv, :][:, sv]))


clf = SVM(C=2, gamma=2)
clf.decomp_method(X_train, y_train, q_value=20, num_iter=1000)
print(clf.accuracy(X_test, y_test))
exit()
# %%
# clf = SVM(C=2,gamma=2)
# clf.decomp_method(X_train, y_train, q_value=800, num_iter=1000)
# print(clf.accuracy(X_test, y_test))

# # len(X_train[clf.a.flatten() >0, :])
# # clf.a#[None, :]
# # len(clf.a * clf.sv_y)

# clf2 = SVM(C=2,gamma=2)
# clf2.fit(X_train, y_train)
# print(clf2.accuracy(X_test, y_test))


from joblib import Parallel, delayed


def get_svm(c, g, q, i, ker):
    clf = SVM(C=c, gamma=g, kernel=ker)
    clf.decomp_method(X_train, y_train, q_value=q, num_iter=i)
    return clf.accuracy(X_test, y_test)


r = range(2, 100, 10)
out = {}
gam = (2, 10,)
j = 0

import matplotlib.pyplot as plt

for k in (polynomial_kernel,):  # , rbf_kernel,):
    out[k.__name__] = Parallel(n_jobs=-1, verbose=10) \
        (delayed(get_svm)(c=2, g=gam[j], q=x, i=1000, ker=k) for x in r)
    j += 1
    plt.plot(list(r), out[k.__name__])
    plt.title(f'{k.__name__.title()} Kernel')
    plt.ylim(0.6, 1)
    plt.show()


# %%
def moving_average(x, k):
    return np.convolve(x, np.ones(k), 'valid') / k


k = 21
plt.plot(list(r[int(k / 2):-int(k / 2)]), moving_average(out, k))
plt.show()


# %%
def get_svm2(c, g, q, i, tol):
    clf = SVM(C=c, gamma=g)
    clf.decomp_method(X_train, y_train, q_value=q, num_iter=i)
    return clf.accuracy(X_test, y_test)


r = range(2, 100, 10)
out = Parallel(n_jobs=-1, verbose=10) \
    (delayed(get_svm2)(c=2, g=2, q=2, i=1000, tol=x) for x in r)
