import numexpr as ne
from Project_2_dataExtraction import *
from cvxopt import matrix
from cvxopt import solvers
import time

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


a = np.ones((2000,))
b = 0.5
N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8 * N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)


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
        
        
    def decomp_method(self, X, y, q_value=10, num_iter=10):
        nfev = 0
        best_loss = 1000

        n_samples, n_features = X.shape
        alpha_k = np.zeros(n_samples)
        alpha_k1 = alpha_k
        patience = 0
        K = self.kernel(X, X, self.gamma)
        t = time.time()

        grad_q = -np.ones_like(y)
        L_neg = set(alpha_k[(alpha_k < 1e-5) & (y==-1)])
        L_pos = set(alpha_k[(alpha_k < 1e-5) & (y==1)])
        U_neg = set(alpha_k[(alpha_k==self.C) & (y==-1)])
        U_pos = set(alpha_k[(alpha_k==self.C) & (y==1)])
        SV = set(alpha_k[(alpha_k > 1e-5) & (alpha_k < self.C)])
        R = L_neg.union(U_pos).union(SV)
        S = L_pos.union(U_neg).union(SV)
        m_alpha = max((-grad_q*y)[list(R)])
        M_alpha = min((-grad_q*y)[list(S)])
        diff = m_alpha - M_alpha
        idx = 0
        while (diff > 0) and idx < num_iter:
            
            W_index = sorted((-grad_q*y)[list(R)], reverse=True)[:q_value//2] + \
                      sorted((-grad_q*y)[list(S)], reverse=True)[:q_value//2]
            
            # Gram matrix
            P = matrix(np.outer(y[W_index], y[W_index]) * K[W_index, :][:, W_index])
            q = matrix(np.ones(q_value) * -1)
            A = matrix(y[W_index], (1, q_value))
            b = matrix(0.0)
            G = matrix(np.vstack((-1 * np.identity(q_value), np.identity(q_value))))
            h = matrix(np.vstack((np.zeros((q_value, 1)), self.C * np.ones((q_value, 1)))))
    
    
            # solve QP sub-problem
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            #self.opt_sol = solution
            nfev += solution['iterations']

            a = np.array(solution['x']).flatten()
            alpha_k1[W_index] = a
            
            grad_q = grad_q + P.dot(alpha_k1-alpha_k) # updating gradient
            alpha_k = alpha_k1
            L_neg = set(alpha_k[(alpha_k < 1e-5) & (y==-1)])
            L_pos = set(alpha_k[(alpha_k < 1e-5) & (y==1)])
            U_neg = set(alpha_k[(alpha_k==self.C) & (y==-1)])
            U_pos = set(alpha_k[(alpha_k==self.C) & (y==1)])
            SV = set(alpha_k[(alpha_k > 1e-5) & (alpha_k < self.C)])
            R = L_neg.union(U_pos).union(SV)
            S = L_pos.union(U_neg).union(SV)
            m_alpha = max((-grad_q*y)[list(R)])
            M_alpha = min((-grad_q*y)[list(S)])
            diff = m_alpha - M_alpha
            
# =============================================================================
#             sv_ind = alpha_k1 > 1e-5
#             a = alpha[sv_ind]
#             sv = X[sv_ind]
#             sv_y = y[sv_ind]
#             ind = np.arange(len(alpha))[sv_ind]
#             b = 0
#             for n in range(len(a)):
#                 b += sv_y[n]
#                 b -= np.sum(a * sv_y * K[ind[n], sv_ind])
#             b /= len(a)
# =============================================================================

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
        self.fit_time = time.time() - t
        self.nfev = nfev
        print(alpha_k)

clf = SVM(C=10,gamma=5)
clf.decomp_method(X_train, y_train, q_value=800, num_iter=500)
print(clf.accuracy(X_test, y_test))