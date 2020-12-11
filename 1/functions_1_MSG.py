import numexpr as ne
from sklearn.svm import SVC
from Project_2_dataExtraction import *
import cvxopt
from cvxopt import matrix
seed = 1679838
np.random.seed(seed)

def rbf_kernel(X, z, gamma):
    X_norm = np.sum(X ** 2, axis=-1)
    y_norm = np.sum(z ** 2, axis=-1)
    return ne.evaluate('exp( -gamma * (A + B - 2 * C))', {
        'A': X_norm[:, None],
        'B': y_norm[None, :],
        'C': np.dot(X, z.T),
        'gamma': gamma
    })

def poly_kernel(X, z, gamma):
    return (X.dot(z.T)+1)**gamma




full_data = np.vstack((xLabel3, xLabel8))
data_norm = full_data/255 # (2000x784)

full_labels = np.hstack((-np.ones((1000,)), np.ones((1000,)))) # (2000,)
kern = {'poly': poly_kernel, 'RBF': rbf_kernel}

def dec_fun(X, y, a, b, kernel='RBF', gamma=1):

    if kernel == 'poly':
        assert gamma >= 1, 'gamma for polynomial kernel must be >= 1'
    elif kernel == 'RBF':
        assert gamma > 0, 'gamma for RBF kernel must be > 1'
    else:
        raise Exception(f'kernel {kernel} not supported. Available kernel types:'
                        f'{list(kern.keys())}')
    k = kern[kernel]
    return np.sign((a[None,:]*y[None,:]).dot(k(X, gamma)) + b)

a = np.ones((2000,))
b=0.5
N = len(data_norm)
indices_train = np.random.choice(list(range(N)), int(0.8*N), replace=False)
X_train = data_norm[indices_train]
X_test = np.delete(data_norm, indices_train, axis=0)
y_train = full_labels[indices_train]
y_test = np.delete(full_labels, indices_train, axis=0)

diag_y_train = y_train * np.identity(len(y_train))
P = matrix(diag_y_train.dot(rbf_kernel(X_train, X_train, 0.05).dot(diag_y_train)), tc='d')
q = matrix(np.ones(y_train.shape[0]), tc='d')
G = matrix(np.vstack([np.identity(X_train.shape[0]), -np.identity(X_train.shape[0])]), tc='d')
C = np.ones(X_train.shape[0])
h = matrix(np.hstack([C, np.zeros(X_train.shape[0]).T]), tc='d')
A = matrix(diag_y_train, tc='d')
b = matrix(np.zeros_like(y_train), tc='d')

from cvxopt import solvers

dic = solvers.qp(P, q, G, h, A, b)
print(dic)
# classifier = SVC(kernel='precomputed')
#
# classifier.fit(poly_kernel(X_train, X_train, 100), y_train)
# classifier.predict(poly_kernel(X_test, X_train, 100))

