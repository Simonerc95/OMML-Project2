import numexpr as ne
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.pardir)

from Project_2_dataExtraction import *
from matplotlib import pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
import time
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

 #
 # def polynomial_kernel(X, z, gamma):
 #    assert gamma >= 1, 'gamma for polynomial kernel must be >= 1'
 #    return (X.dot(z.T) + 1) ** gamma
def polynomial_kernel(X, z, gamma):
    assert gamma >= 1, 'gamma for polynomial kernel must be >= 1'
    return (X.dot(z.T)) ** gamma

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

    def fit(self, X, y):
        start = time.time()
        n_samples, n_features = X.shape
        tol = 1e-6
        # Gram matrix
        K = self.kernel(X, X, self.gamma)

        P = matrix((np.outer(y, y) * K))
        q = matrix(np.ones(n_samples) * -1)
        A = matrix(y.dot(np.identity(len(y))), (1, n_samples))
        b = matrix(0.0)
        G = matrix(np.vstack((-1*np.identity(n_samples), np.identity(n_samples))))
        h = matrix(np.vstack((np.zeros((n_samples, 1)), self.C*np.ones((n_samples, 1)))))

        # solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        self.opt_sol = solution
        self.alpha = np.array(solution['x']).flatten()
        a = self.alpha

        # Support vectors have non zero lagrange multipliers
        sv = a > tol
        print(sum(sv))
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
        self.cpu_time = time.time() - start

        # Weight vector

    def predict(self, X):
        k = self.kernel
        return np.sign((self.a[None, :] * self.sv_y[None, :]).dot(k(X, self.sv, self.gamma).T) + self.b)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

    def get_objective(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = self.kernel(X, X, self.gamma)
        P = matrix(np.outer(y, y) * K)
        q = np.ones((1, n_samples))
        alpha = np.array(self.alpha)
        obj = 0.5*(alpha.T.dot(P).dot(alpha)) - q.dot(alpha)
        return obj

    def output(self):
        print(self.opt_sol)
        
    def conf_mat(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y, y_pred)
        return cm



def k_fold(X_train, y_train, C=10, gamma=2, folds=4):
    """
    K-fold cross-validation algorithm to perform gridsearch
    on the hyperparameters gamma and C
    """

    N = len(X_train)
    shuffle_indices = np.random.choice(list(range(N)), N, replace=False)
    shuffled_train = X_train[shuffle_indices]
    shuffled_labels = y_train[shuffle_indices]
    n_val = int(len(X_train)/folds)

    accuracies = []
    for k in range(folds):
        indices_valid = list(range(k*n_val, (k+1)*n_val))
        valid = shuffled_train[indices_valid]
        train = np.delete(shuffled_train, indices_valid, axis=0)
        valid_labs = shuffled_labels[indices_valid]
        train_labs = np.delete(shuffled_labels, indices_valid, axis=0)
        clf = SVM(kernel=polynomial_kernel, gamma=gamma, C=C)
        clf.fit(train, train_labs)
        accuracies.append(clf.accuracy(valid, valid_labs))
    accuracy = np.mean(accuracies)

    return accuracy

#
# def GridSearch(X_train, y_train, X_test, y_test, L_C=1, U_C=10, L_gamma=1, U_gamma=10):
#     start = time.time()
#
#     gammas = range(L_gamma, U_gamma+1)
#     Cs = range(L_C, U_C+1)
#     train_accs = np.zeros((len(Cs), len(gammas)))
#     test_accs = np.zeros_like(train_accs)
#     best_acc = 0
#     best_params = {'C':None, 'gamma':None}
#     for i, ci in enumerate(Cs):
#         for j, g in enumerate(gammas):
#             acc = k_fold(X_train, y_train, C=ci, gamma=g, folds=4)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_params['C'] = ci
#                 best_params['gamma'] = g
#             clf = SVM(kernel=polynomial_kernel, gamma=g, C=ci)
#             clf.fit(X_train, y_train)
#             train_accs[i][j] = clf.accuracy(X_train, y_train)
#             test_accs[i][j] = clf.accuracy(X_test, y_test)
#
#     plot3D(Cs, gammas, train_accs, test_accs)
#     print(f'elapsed time: {round(time.time() - start,2)}')
#     return best_params, train_accs, test_accs
#




#best_params, train_accs, test_accs = GridSearch(X_train, y_train, X_test, y_test, U_C=2, U_gamma=2)
#print(best_params)
#print(train_accs)
#print(test_accs)
cl = SVM(kernel=polynomial_kernel, C=10, gamma=2)
cl.fit(X_train, y_train)
print(cl.accuracy(X_test,  y_test))
print(cl.get_objective(X_train,  y_train))
cl.output()
