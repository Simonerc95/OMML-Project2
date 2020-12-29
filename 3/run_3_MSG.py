from functions_3_MSG import *

used_kernel = polynomial_kernel
best_params = {'C': 0.01, 'gamma': 2}
q_value = 2

cl = SVM(kernel=used_kernel, **best_params)
cl.linesearch(X_train, y_train, q_value=q_value, num_iter=10000)
train_acc = cl.accuracy(X_train, y_train)
test_acc = cl.accuracy(X_test, y_test)
cm = cl.conf_mat(X_test, y_test)
n_iters = cl.iterations

print(f'C = {best_params["C"]}, gamma = {best_params["gamma"]}, {used_kernel.__name__}')
print(f'Classification rate on the training set = {train_acc}')
print(f'Classification rate on the test_set = {test_acc}')
print(f'Counfusion matrix: \n {cm}')
print(f'Time for the optimization: {cl.fit_time}')
print(f'Num iter: {n_iters}')
print(f'Difference m(alpha) - M(alpha): {cl.diff}')
