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
        self.diff = m_alpha - M_alpha


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

    train_accuracies = []
    val_accuracies = []
    for k in range(folds):
        indices_valid = list(range(k*n_val, (k+1)*n_val))
        valid = shuffled_train[indices_valid]
        train = np.delete(shuffled_train, indices_valid, axis=0)
        valid_labs = shuffled_labels[indices_valid]
        train_labs = np.delete(shuffled_labels, indices_valid, axis=0)
        clf = SVM(kernel=polynomial_kernel, gamma=gamma, C=C)
        try:
            clf.fit(train, train_labs)
            val_accuracies.append(clf.accuracy(valid, valid_labs))
            train_accuracies.append(clf.accuracy(train, train_labs))
        except:
            print(f'Wrong Parameters C = {C}, gamma = {gamma}')
            val_accuracies.append(0)
            train_accuracies.append(0)
            break
    train_accuracy = np.mean(train_accuracies)
    val_accuracy = np.mean(val_accuracies)
    return train_accuracy, val_accuracy

def plot3D(Cs, gammas, train_accs, test_accs):

    x_1, x_2 = np.meshgrid(Cs, gammas)
    z_1 = train_accs
    z_2 = test_accs
    fig = plt.figure(figsize=(8, 6))

    ax = plt.axes(projection='3d')
    ax.plot_surface(x_1, x_2, z_1.reshape(x_1.shape), rstride=1, cstride=1,
                    cmap='Blues', edgecolor='none')
    ax.plot_surface(x_1, x_2, z_2.reshape(x_1.shape), rstride=1, cstride=1,
                    cmap='Reds', edgecolor='none')
    ax.set_title('surface')
    plt.savefig('out_1', dpi=100)

def GridSearch(X_train, y_train, L_C=0.1, U_C=1, L_gamma=1, U_gamma=2):
    start = time.time()

    gammas = range(L_gamma, U_gamma+1)
    Cs = np.arange(L_C, U_C+1, 0.2)
    train_accs = np.zeros((len(Cs), len(gammas)))
    val_accs = np.zeros_like(train_accs)
    best_acc = 0
    best_params = {'C':None, 'gamma':None}
    for i, ci in enumerate(Cs):
        for j, g in enumerate(gammas):
            train_acc, val_acc = k_fold(X_train, y_train, C=ci, gamma=g, folds=4)
            if val_acc > best_acc:
                best_acc = val_acc
                best_params['C'] = ci
                best_params['gamma'] = g
            train_accs[i][j] = train_acc
            val_accs[i][j] = val_acc

    plot3D(Cs, gammas, train_accs, val_accs)
    print(f'elapsed time: {round(time.time() - start,2)}')
    return best_params, train_accs, val_accs

