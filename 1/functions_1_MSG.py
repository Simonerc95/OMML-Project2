import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import product
import numexpr as ne
from Project_2_dataExtraction import *

seed = 1679838
np.random.seed(seed)

def rbf_kernel(X, gamma):
    X_norm = np.sum(X ** 2, axis=-1)
    y_norm = np.sum(X ** 2, axis=-1)
    return ne.evaluate('exp( -gamma * (A + B - 2 * C))', {
        'A': X_norm[:, None],
        'B': y_norm[None, :],
        'C': np.dot(X, X.T),
        'gamma': gamma
    })

def poly_kernel(X, gamma):
    return (X.dot(X.T)+1)**gamma




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

class MLP:
    def __init__(self, df, N, rho, sigma=1, ttv=[.7, .85], act_func='hyperbolic_tangent', random_state=1679838):

        assert N > 0, 'N must be positive!'
        assert isinstance(N, int), 'N must be an integer'
        self.N = N

        assert sigma >= 0, 'Sigma must be positive!'
        self.sigma = sigma

        self.rho = rho

        assert act_func in activations.keys(), f'"{act_func}" is a not supported activation function'
        self.activation = activations[act_func]

        self.seed = random_state
        self.neurons = []
        np.random.seed(self.seed)
        self.train_data, self.test_data, self.valid_data = np.split(df.sample(frac=1),
                                                                    [int(ttv[0] * len(df)), int(ttv[1] * len(df))])

        self.X_train = self.train_data.iloc[:, :2].to_numpy()  # P x n
        self.y_train = self.train_data.iloc[:, 2].to_numpy()  # P x 1
        self.X_valid = self.valid_data.iloc[:, :2].to_numpy()
        self.y_valid = self.valid_data.iloc[:, 2].to_numpy()
        self.X_test = self.test_data.iloc[:, :2].to_numpy()
        self.y_test = self.test_data.iloc[:, 2].to_numpy()

        self.n = self.X_train.shape[1]
        self.W = np.random.normal(scale=2, size=(self.N, self.n))  # N x n
        self.v = np.random.normal(scale=2, size=(self.N, 1))  # N x 1
        self.b = np.random.normal(scale=2, size=(self.N, 1))  # N x 1
        self.Loss_list = []

    def fit(self, method='bfgs', maxiter=1000, print_=True):
        disp = print_
        self.method = method
        self.max_iter = maxiter
        t = time.time()
        vec = self._to_vec()

        opt = minimize(self._optimize, vec, method=method, options=
        {'maxiter': maxiter, 'disp': disp})
        self.minimize_obj = opt
        self.W, self.v, self.b = self._to_array(opt.x)
        self._get_all_loss()
        self.fit_time = time.time() - t
        if print_:
            pass
            print(f'Time: {self.fit_time}')
            print(f'Loss_train_reg_fit from minimize:{self.minimize_obj["fun"]}')
            print(f'Loss_valid :{self.valid_loss}')

    def _compute_loss(self, W, v, b, dataset, loss_reg=False):
        sigma = self.sigma
        rho = self.rho
        activation = self.activation

        if dataset == 'valid':
            X = self.X_valid
            y = self.y_valid
        elif dataset == 'train':
            X = self.X_train
            y = self.y_train
        elif dataset == 'test':
            X = self.X_test
            y = self.y_test

        xx = W.dot(X.T) - b  # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = activation(xx, sigma)  # , sigma=self.sigma) # N x P
        f_x = g_x.T.dot(v)  # P x 1 - g_x.T = P x N @ N x 1
        Loss = np.sum((f_x.reshape(y.shape) - y) ** 2) / (2 * len(y))
        # self.Loss_list.append(Loss)

        if loss_reg:
            L2 = np.linalg.norm(np.concatenate((W, v, b), axis=None)) ** 2  # regularization
            Loss_reg = Loss + (rho * L2)
            return Loss_reg
        else:
            return Loss

    def predict(self, X):
        xx = self.W.dot(
            X.T) - self.b  # N x P Ogni colonna è l'output degli N neuroni per una specifica x (unità statistica)
        g_x = self.activation(xx)  # , sigma=self.sigma) # N x P
        f_x = g_x.T.dot(self.v)  # P x 1 - g_x.T = P x N @ N x 1

        return f_x

    def _to_vec(self):
        return np.hstack([self.W.flatten(), self.v.flatten(), self.b.flatten()])

    def _to_array(self, vec):
        N = self.N
        n = self.n

        assert vec.shape == (N * n + (2 * N),)
        return vec[:N * n].reshape(N, n), vec[N * n:-N].reshape(N, 1), vec[-N:].reshape(N, 1)

    def _optimize(self, vec, dataset='train'):
        W, v, b = self._to_array(vec)
        return self._compute_loss(W, v, b, dataset, loss_reg=True)

    def get_loss(self, loss_type):
        out = {}
        if loss_type == 'all':
            for type_ in ('train', 'valid', 'test',):
                out[type_] = self._compute_loss(self.W, self.v, self.b, dataset=type_)
        else:
            out[loss_type] = self._compute_loss(self.W, self.v, self.b, dataset=loss_type)

        return out

    def _get_all_loss(self):
        self.train_loss = self._compute_loss(self.W, self.v, self.b, dataset='train', loss_reg=False)
        self.valid_loss = self._compute_loss(self.W, self.v, self.b, dataset='valid', loss_reg=False)
        self.test_loss = self._compute_loss(self.W, self.v, self.b, dataset='test', loss_reg=False)
        self.train_loss_reg = self._compute_loss(self.W, self.v, self.b, dataset='train', loss_reg=True)

    def print_loss_param(self, time=True):
        opt = self.minimize_obj
        # if time:
        #     print(f'\n', self.fit_time)
        print('\nBest N :', self.N,
              '\nBest sigma :', self.sigma,
              '\nBest rho :', self.rho,
              '\nMax iterations:', self.max_iter,
              '\nOptimization solver:', self.method,
              '\nNumber of function evaluations:', opt['nfev'],
              '\nNumber of gradient evaluations:', opt['njev'],
              '\nTime for optimizing the network:', self.fit_time,
              '\nBest train_loss: ', self.train_loss,
              # '\nBest valid_loss: ', self.valid_loss,
              '\nBest test_loss: ', self.test_loss, )


params = {
    'N_vals': list(range(1, 50, 1)),
    'sigma_vals': np.arange(.5, 1.5, .1),
    'rho_vals': [1e-5, 1e-3, 1e-4]}


def random_search(model, df, params, iterations=1000, seed=1679838, print_=True, n_jobs=-1):
    np.random.seed(seed)
    combinations = np.array(list(product(*params.values())))
    np.random.shuffle(combinations)
    combinations = combinations[:iterations]
    assert iterations <= len(combinations), 'iterations exceeded number of combinations'
    t = time.time()  # x[0] = N, x[1] = sigma, x[2] = rho
    res = Parallel(n_jobs=n_jobs, verbose=10) \
        (delayed(get_opt)(model, int(x[0]), x[1], x[2], df, print_) for x in combinations)
    print(f"\nTotal time: {time.time() - t}")
    best_loss = np.inf
    model = None
    for mod in res:
        if mod.valid_loss < best_loss:
            model = mod

    return model


def get_opt(model, n, sigma, rho, df, print_=True):
    network = model(df=df, N=n, rho=rho, sigma=sigma)
    network.fit(print_=print_)
    return network


def get_plot(net):
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)

    x_1, x_2 = np.meshgrid(x, y)
    x_1 = x_1.flatten()  # .reshape(-1,1)
    x_2 = x_2.flatten()  # .reshape(-1,1)
    x_ = np.vstack([x_1, x_2])
    z_ = net.predict(x_.T)

    fig = plt.figure(figsize=(8, 6))

    ax = plt.axes(projection='3d')
    x_1, x_2 = np.meshgrid(x, y)
    ax.plot_surface(x_1, x_2, z_.reshape(x_1.shape), rstride=1, cstride=1,
                    cmap='gist_rainbow_r', edgecolor='none')
    ax.set_title('surface')
    plt.savefig('out_11', dpi=100)


def get_overfitting_plots(Network, df):
    params = dict(
        N=list(range(1, 72, 5)),
        sigma=np.arange(0.5, 1.5, 0.1),
        rho=np.arange(1e-5, 1.05e-3, 5e-5)
    )
    base = dict(N=28, rho=1e-4, sigma=0.7)
    # for par in params.keys():
    par = 'rho'
    print(f'{par}')
    train_losses = []
    valid_losses = []
    for i, _ in enumerate(params[par]):
        N = params[par][i] if par == 'N' else base['N']
        sigma = params[par][i] if par == 'sigma' else base['sigma']
        rho = params[par][i] if par == 'rho' else base['rho']

        net = Network(df, N=N, rho=rho, sigma=sigma)
        net.fit(print_=False)
        net._get_all_loss()
        train_losses.append(net.train_loss)
        valid_losses.append(net.valid_loss)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(params[par], train_losses, label='Train Loss')
    plt.plot(params[par], valid_losses, label='Valid Loss')
    plt.xlabel(f'{par}')
    plt.ylabel('Loss')
    plt.title(f"Loss trend on {par}")
    plt.legend()
    plt.savefig(f'{par}_11_losses', dpi=100)
    plt.clf()