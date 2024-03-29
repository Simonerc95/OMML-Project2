from functions_1_MSG import *

gridsearch = False
used_kernel = polynomial_kernel
best_params = {'C': 0.01, 'gamma': 2}

if gridsearch:
    cc = np.power(10, np.arange(-5., 2.))
    cc = np.concatenate([cc, np.arange(1.,11.)])
    gg = list(range(1, 4))

    best_params = GridSearch(X_train, y_train, cc, gg, kernel=used_kernel)


cl = SVM(kernel=used_kernel, **best_params)
cl.fit(X_train, y_train)
train_acc = cl.accuracy(X_train, y_train)
test_acc = cl.accuracy(X_test, y_test)
cm = cl.conf_mat(X_test, y_test)
n_iter = cl.opt_sol['iterations']

print(f'C = {best_params["C"]}, gamma = {best_params["gamma"]}, {used_kernel.__name__}')
print(f'Classification rate on the training set = {train_acc}')
print(f'Classification rate on the test_set = {test_acc}')
print(f'Counfusion matrix: \n {cm}')
print(f'Time for the optimization: {cl.cpu_time}')
print(f'Num iter: {n_iter}')
print(f'Difference m(alpha) - M(alpha): {cl.diff}')