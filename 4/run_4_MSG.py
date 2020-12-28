from functions_4_MSG import *

used_kernel = polynomial_kernel
best_params = {'C': 2, 'gamma': 2}

cl = SVMMulticlass(kernel=used_kernel, **best_params)
cl.fit_multi(X_train, y_train)
train_acc = cl.accuracy_multi(X_train, y_train)
test_acc = cl.accuracy_multi(X_test, y_test)
cm = cl.conf_mat(X_test, y_test)
n_iters = cl.iter
max_diff = max(cl.diff.values())

print(f'C = {best_params["C"]}, gamma = {best_params["gamma"]}, {used_kernel.__name__}')
print(f'Classification rate on the training set = {train_acc}')
print(f'Classification rate on the test_set = {test_acc}')
print(f'Counfusion matrix: \n {cm}')
print(f'Time for the optimization: {cl.fit_time}')
print(f'Num iter: {n_iters}')
print(f'Difference m(alpha) - M(alpha): {max_diff}')