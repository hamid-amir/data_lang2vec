import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV


seed = 8446
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
}

base_estimator = RandomForestClassifier(random_state=seed, n_jobs=28)

train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names = pickle.load(open('feats.pickle', 'rb'))

sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=2, resource='n_estimators',
                         max_resources=30).fit(X, y)

print(sh.best_estimator_)
print(sh.best_score_)
