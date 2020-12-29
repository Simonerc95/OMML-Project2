import numexpr as ne
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append(os.pardir)

from Project_2_dataExtraction import *
from joblib import Parallel, delayed
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
import time
from sklearn.metrics import confusion_matrix
import numpy as np

seed = 1679838
np.random.seed(seed)


def rbf_kernel(X, z, gamma):
    '''
    Used for comparison, but not for final evaluations
    '''
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
    '''
    Kernel used for whole project
    '''
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

        P = matrix((np.outer(y, y) * K), tc='d')
        q = matrix(np.ones(n_samples) * -1, tc='d')
        A = matrix(y.dot(np.identity(n_samples)), (1, n_samples), tc='d')
        b = matrix(0.0, tc='d')
        G = matrix(np.vstack((-1*np.identity(n_samples), np.identity(n_samples))), tc='d')
        h = matrix(np.vstack((np.zeros((n_samples, 1)), self.C*np.ones((n_samples, 1)))), tc='d')

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

        # Evaluation of objective function
        # e = np.ones_like(a)
        # print(f'dual objective  with formula',
        #       0.5 * (a.T.dot(P).dot(a)) - e.T.dot(a))


    def predict(self, X):
        k = self.kernel
        return np.sign((self.a[None, :] * self.sv_y[None, :]).dot(k(X, self.sv, self.gamma).T) + self.b)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)
        
    def conf_mat(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.flatten()
        cm = confusion_matrix(y, y_pred)
        return cm


def k_fold(X_train, y_train, kernel, C=10, gamma=2, folds=4):
    """
    K-fold cross-validation algorithm to perform gridsearch
    on the hyperparameters gamma and C
    """

    N = len(X_train)
    shuffle_indices = np.random.choice(list(range(N)), N, replace=False)
    shuffled_train = X_train[shuffle_indices]
    shuffled_labels = y_train[shuffle_indices]

    # taking validation folds of 1600/4 = 400 observations (remaining 1200 are used for training)
    n_val = int(len(X_train)/folds)

    train_accuracies = []
    val_accuracies = []
    for k in range(folds):
        indices_valid = list(range(k*n_val, (k+1)*n_val))
        valid = shuffled_train[indices_valid]
        train = np.delete(shuffled_train, indices_valid, axis=0)
        valid_labs = shuffled_labels[indices_valid]
        train_labs = np.delete(shuffled_labels, indices_valid, axis=0)
        clf = SVM(kernel=kernel, gamma=gamma, C=C)
        clf.fit(train, train_labs)
        val_accuracies.append(clf.accuracy(valid, valid_labs))
        train_accuracies.append(clf.accuracy(train, train_labs))

    train_accuracy = np.mean(train_accuracies)
    val_accuracy = np.mean(val_accuracies)
    return train_accuracy, val_accuracy


def get_performance(X_train, y_train, C, gamma, kernel, folds=4):
    '''
    Used for parallelization of GridSearch
    '''
    train_acc, val_acc = k_fold(X_train, y_train, kernel=kernel, C=C, gamma=gamma, folds=folds)
    return {"C": C, "gamma": gamma, "train_accuracy": train_acc, "validation_accuracy": val_acc}

def GridSearch(X_train, y_train, C_list, gamma_list, kernel = polynomial_kernel, n_jobs=-1, verbose=10):
    start = time.time()
    data = Parallel(n_jobs=n_jobs, verbose=verbose)\
                (delayed(get_performance)(X_train, y_train, C=c, gamma=g, folds=4, kernel=kernel) for c in C_list for g in gamma_list)
    df = pd.json_normalize(data)
    best = df.iloc[df.validation_accuracy.argmax()]

    print(f'elapsed time: {round(time.time() - start,2)}')
    best_params = best[["C", "gamma"]].to_dict()
    return best_params

