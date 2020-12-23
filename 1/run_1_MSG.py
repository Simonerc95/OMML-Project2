from functions_1_MSG import *

gridsearch = False
best_params = {'C': 2, 'gamma': 2}

if gridsearch:
    best_params, train_accs, test_accs = GridSearch(X_train, y_train, X_test, y_test, U_C=2, U_gamma=2)

cl = SVM(kernel=polynomial_kernel, **best_params)
cl.fit(X_train, y_train)
train_acc = cl.accuracy(X_train, y_train)
test_acc = cl.accuracy(X_test, y_test)
cm = cl.conf_mat(X_test, y_test)
n_iter = cl.opt_sol['iterations']
print(f'C = {best_params["C"]}, gamma = {best_params["gamma"]}')
print(f'Classification rate on the training set = {train_acc}')
print(f'Classification rate on the test_set = {test_acc}')
print(f'Counfusion matrix: \n {cm}')
print(f'Num iter: {n_iter}')