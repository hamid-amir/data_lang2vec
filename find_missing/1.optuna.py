import pickle
import time
import sys
import os
import json
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(os.getcwd())
import scripts.myutils as myutils
from find_missing.utils import cross_validation, get_clf



def objective(trial, method: str):
    features = ['lang_id', 'feat_id', 'geo_lat', 'geo_long', 'lang_group', 'aes_status', 'wiki_size', 'num_speakers', 'lang_fam', 'scripts', 'feat_name', 'phylogency']
    remove_features = [trial.suggest_categorical(feature, [True, False]) for feature in features]
    remove_features = [feature for feature, selected in zip(features, remove_features) if selected]
    if len(remove_features) == len(features):
        remove_features = []
    n_components = trial.suggest_int('n_components', 10, 100)

    # construct dataset using 300 langs, based on these settings
    myutils.extract_features(classifier='find_missing', n_components=n_components, n=300, remove_features=remove_features)
    x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))

    if method == 'svm':  # It's too time-consuming and therefore not practical. (Not recommended)
        C = trial.suggest_float('C', 1e-4, 1e3, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 1, 10)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    elif method == 'rf':
        max_depth = trial.suggest_int('max_depth', 3, 1000)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"])
        class_weight = trial.suggest_categorical('class_weight', [None, "balanced", "balanced_subsample"])
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        clf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                     criterion=criterion, class_weight=class_weight, n_estimators=n_estimators)
    elif method == 'logres':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
        C = trial.suggest_float('C', 1e-4, 1e3, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        clf = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver)

    elif method == 'knn':
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 2) 
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    
    elif method == 'gbc':  # Takes a long time (but seems to have better performance) so we used only 10 iters for it
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

    elif method == 'dt': 
        max_depth = trial.suggest_int('max_depth', 1, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

    try:
        result = cross_validation(clf, x, y)
        return result['f1']
    except ValueError:
        return 0

    


def hyperparameter_tuning(method: str, n_trails: int, save_dir: str = 'result'):
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, method), n_trials=n_trails)

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f'{save_dir}/optimization_history_{method}.png')
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f'{save_dir}/slice_{method}.png')

    results = []
    for trial in study.trials:
        trial_result = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state)
        }
        results.append(trial_result)

    # Convert results to JSON format
    results_json = json.dumps(results, indent=4)

    # Save JSON data to a file
    with open(f'{save_dir}/optuna_study_results_{method}.json', 'w') as f:
        f.write(results_json)

    best_trial = study.best_trial
    print(f'Best trial: {best_trial}')
    seconds = time.time() - start_time
    print('seconds: {:.2f}'.format(seconds))
    return best_trial.values


if __name__ == '__main__':
    method = sys.argv[1]
    n_trials = int(sys.argv[2])   # used 10 iters for gbc and 100 iters for other classifiers
    # settings = {'rf': 100, 'logres': 100, 'knn': 100, 'gbc': 10, 'dt': 100}
    # for method, n_trials in settings.items():
    hyperparameter_tuning(method, n_trials)
