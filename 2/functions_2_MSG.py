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
        tol = 1e-5
        y = y.reshape(1600,1)

        n_samples, n_features = X.shape
        alpha_k = np.zeros(y.shape)
      
        alpha_k1 = np.zeros(y.shape)

        K = self.kernel(X, X, self.gamma)
        K = np.array(K)
        t = time.time()
   
        grad_q = -np.ones_like(y)
        L_neg = np.where((alpha_k < tol) & (y==-1))[0]
        L_pos = np.where((alpha_k < tol) & (y==1))[0]
        U_neg = np.where((alpha_k >= self.C - tol) & (y==-1))[0]
        U_pos = np.where((alpha_k >= self.C - tol) & (y==1))[0]
        SV = np.where((alpha_k > tol) & (alpha_k < self.C))[0]
        S = np.array(L_neg.tolist() + U_pos.tolist() + SV.tolist())
        R = np.array(L_pos.tolist() + U_neg.tolist() + SV.tolist())
        #assert len(R) == len(set(R)), 'doubles'
        m_alpha = max((-grad_q*y)[R,])
        #print('m_alpha', m_alpha)
        M_alpha = min((-grad_q*y)[S,])
        #print('M_alpha', M_alpha)
        diff = m_alpha - M_alpha
        #print('diff', diff)
        idx = 0
        while (diff > 0) and idx < num_iter:
            
            #W_index = R[np.argsort((-grad_q*y)[R,], axis=0)[:q_value//2][::-1]].flatten().tolist() + \
                      #S[np.argsort((-grad_q*y)[S,], axis=0)[:q_value//2]].flatten().tolist() 
                      
            W_index = np.random.choice(R,5).tolist() + np.random.choice(S,5).tolist()
            #print('grad_S',np.argsort((-grad_q*y)[R,], axis=0)[:q_value//2])
            #print('w_index', W_index)
            
            # Gram matrix
            #print('outer', np.outer(y[W_index,], y[W_index,]).shape)
            #print('k', K[W_index, :][:, W_index].shape)
            P = matrix(np.outer(y[W_index,], y[W_index,]) * K[W_index, :][:, W_index])
            q = matrix(np.ones(q_value) * -1)
            A = matrix(y[W_index,], (1, q_value))
            b = matrix(0.0)
            G = matrix(np.vstack((-1 * np.identity(q_value), np.identity(q_value))))
            h = matrix(np.vstack((np.zeros((q_value, 1)), self.C * np.ones((q_value, 1)))))
    
    
            # solve QP sub-problem
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            #self.opt_sol = solution
            nfev += solution['iterations']

            a = np.array(solution['x']).flatten()
            #print(a)
            alpha_k1[W_index,] = a.reshape((q_value,1))
            
            #print('graddot', np.array(P).dot(alpha_k1[W_index]-alpha_k[W_index]))
            #print('grad-upt', alpha_k1[W_index]-alpha_k[W_index])
            grad_q[W_index,] = grad_q[W_index,] + (np.array(P).dot(alpha_k1[W_index,]-alpha_k[W_index,])) # updating gradient
            alpha_k[W_index,] = alpha_k1[W_index,]
            L_neg = np.where((alpha_k < tol) & (y==-1))[0]
            L_pos = np.where((alpha_k < tol) & (y==1))[0]
            U_neg = np.where((alpha_k >= self.C - tol) & (y==-1))[0]
            U_pos = np.where((alpha_k >= self.C - tol) & (y==1))[0]
            SV = np.where((alpha_k > tol) & (alpha_k < self.C))[0]
            S = np.array(L_neg.tolist() + U_pos.tolist() + SV.tolist())
            R = np.array(L_pos.tolist() + U_neg.tolist() + SV.tolist())
            #assert len(R) == len(set(R)), 'doubles'
            m_alpha = max((-grad_q*y)[R,])
            M_alpha = min((-grad_q*y)[S,])
            diff = m_alpha - M_alpha
            print(diff)
            
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
        print('iter', idx)
        #print(alpha_k[np.where(alpha_k !=0)[0]])

clf = SVM(C=2,gamma=2)
clf.decomp_method(X_train, y_train, q_value=10, num_iter=100)
#print(clf.accuracy(X_test, y_test))