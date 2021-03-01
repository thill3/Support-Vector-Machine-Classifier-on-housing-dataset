from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'kernel': ["linear"], 'C': [.1, 1, 10]},
    {'kernel': ["rbf"], 'C': [.1, 1, 10], 'gamma': [.1, 1, 10]},
]

sv_reg = SVR()
grid_search = GridSearchCV(sv_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=4, njobs=6)

grid_search.fit(housing_train_prepared, train_labels)
# It will use 5 fold CV on each combo

print(grid_search.best_params_)

print(grid_search.best_estimator_)
# takes about 10 minutes

# because the best estimator is at an end of the range then we adjust the
# range to go beyond that end.
# so we eliminate the rbf line and redo the 'C' options

param_grid1 = [
    {'kernel': ["linear"], 'C': [1 * 10 ** 5, 2 * 10 ** 5, 3 * 10 ** 5]},
]

sv_reg = SVR()
grid_search = GridSearchCV(
    sv_reg,
    param_grid1,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=3,  # higher = more messages
    n_jobs=4  # number of jobs to run in parallel
)

grid_search.fit(housing_train_prepared, train_labels)  # note that what I'm using here are the original labels
# rather than the OneHotEncoded labels

print(grid_search.best_params_)

print(grid_search.best_estimator_)

#C = 10**5 is still the best and the max.
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
print(rmse)

# because the best estimator is at an end of the range then we adjust the
# range to go beyond that end.
# so redo the 'C' options
param_grid2 = [
    {'kernel': ["linear"], 'C': [10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]},
]

sv_reg = SVR()
grid_search = GridSearchCV(
    sv_reg,
    param_grid2,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=3,  # higher = more messages
    n_jobs=4  # number of jobs to run in parallel
)

grid_search.fit(housing_train_prepared, train_labels)

print(grid_search.best_params_)

print(grid_search.best_estimator_)

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
print(rmse)

#################################################################################
