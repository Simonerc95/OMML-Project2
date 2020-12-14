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
        G = matrix(np.diag(np.ones(n_samples) * -1))
        h = matrix(np.zeros(n_samples))


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

    def predict(self, X):
        k = self.kernel
        return np.sign((self.a[None, :] * self.sv_y[None, :]).dot(k(X, self.sv, self.gamma).T) + self.b)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

    def test_loss(self, X, y):
        pass

    def output(self):
        print(self.opt_sol)
        
        
    def decomp_method(self, X, y, q_value=4, num_iter=10, tol=4):
        nfev = 0
        best_loss = 1000
        self.tot_fun = 0
        n_samples, n_features = X.shape
        alpha_upt = np.zeros(n_samples)
        t = time.time()
        for c in range(num_iter):
            W_index = np.random.choice(list(range(n_samples)), q_value, replace=True)
            
            # Gram matrix
            K = self.kernel(X[W_index,:], X, self.gamma)
    
            P = matrix(np.outer(y[W_index,:], y[W_index,:]) * K)
            q = matrix(np.ones(q_value) * -1)
            A = matrix(y, (1, q_value))
            b = matrix(0.0)
            G = matrix(np.diag(np.ones(q_value) * -1))
            h = matrix(np.zeros(q_value))
    
    
            # solve QP problem
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            #self.opt_sol = solution
            nfev += solution['iterations']
            a = np.array(solution['x']).flatten()
            
            alpha_upt[W_index] = a
    
    
            # Support vectors have non zero lagrange multipliers
            sv = alpha_upt > 1e-5
            ind = np.arange(len(alpha_upt))[sv]
            self.a = alpha_upt[sv]
            self.sv = X[sv]
            self.sv_y = y[sv]
            # print("%d support vectors out of %d points" % (len(self.a), n_samples))
    
            # Intercept
            self.b = 0
            for n in range(len(self.a)):
                self.b += self.sv_y[n]
                self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
            self.b /= len(self.a)
            
            self.tot_fun += 1
            current_loss = self._optimize(v)
            if current_loss < best_loss:
                self.W = self.W_tmp
                self.b = self.b_tmp
                self.v = v
                best_loss = current_loss
       
        self.fit_time = time.time() - t
        self.nfev = nfev
