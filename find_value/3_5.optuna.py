import pickle
import time
import sys
import os
import json
import argparse
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier 
sys.path.append(os.getcwd())
import scripts.myutils as myutils
from find_missing.utils import cross_validation, get_clf



def load_data(use_filtered: bool):
    if use_filtered:
        miltale_feats_path = 'miltale_extracted_feats.pickle'
    else:
        miltale_feats_path = 'miltale_extracted_feats_unfiltered.pickle'
    
    miltale_langs, miltale_X_sparse = pickle.load(open(miltale_feats_path, 'rb'))
    miltale_X = miltale_X_sparse.toarray()
    max_feats = min(miltale_X.shape[0], miltale_X.shape[1])
    
    return miltale_X, max_feats


def objective(trial, method: str, n_langs: int, target_feature: str, max_feats: int, use_filtered: bool):
    features = ['lang_id', 'feat_id', 'geo_lat', 'geo_long', 'lang_group', 'aes_status', 'wiki_size', 'num_speakers', 'lang_fam', 'scripts', 'feat_name', 'phylogency']
    remove_features = [trial.suggest_categorical(feature, [True, False]) for feature in features]
    remove_features = [feature for feature, selected in zip(features, remove_features) if selected]
    miltale_data = trial.suggest_categorical('miltale_data', [True, False])
    if len(remove_features) == len(features):
        remove_features = []

    n_components = trial.suggest_int('n_components', 10, 100)
    miltate_n_components = trial.suggest_int('miltate_n_components', 10, max_feats)
    if n_langs == -1: n_langs = None
    myutils.extract_features(
        'find_value',
        n_components=n_components,
        miltate_n_components=miltate_n_components,
        n=n_langs,
        remove_features=remove_features,
        miltale_data=miltale_data,
        job_number=5,
        use_filtered=use_filtered
    )

    X, Y, names, all_feat_names = pickle.load(open('feats-full_find_value_5.pickle', 'rb'))

    new_X, new_Y = list(), list()
    for i, name in enumerate(names):
        if name.split('|')[1] == target_feature:
            new_X.append(X[i])
            new_Y.append(Y[i])

    if method == 'svm':
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
                                     criterion=criterion, class_weight=class_weight, n_estimators=n_estimators, n_jobs=16)
    elif method == 'logres':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)
        C = trial.suggest_float('C', 1e-4, 1e3, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        clf = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver, n_jobs=16)
    elif method == 'xgb':
        eta = trial.suggest_float('eta', 1e-8, 1.0, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 15)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.1, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
        alpha = trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        lambda_ = trial.suggest_float('lambda', 1e-8, 1.0, log=True)
        clf = XGBClassifier(
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=alpha,
            reg_lambda=lambda_,
            eval_metric='mlogloss',
            n_jobs=16
        )
    elif method == 'dt':
        max_depth = trial.suggest_int('max_depth', 1, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, criterion=criterion)
        
    try:
        result = cross_validation(clf, new_X, new_Y)
        return result['f1']
    except ValueError:
        return 0

    
def hyperparameter_tuning(method: str, n_trials: int, n_langs: int,target_feature: str, max_feats: int, save_dir: str = 'result', use_filtered: bool = True):
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, method, n_langs, target_feature, max_feats, use_filtered), n_trials=n_trials)

    optuna.visualization.matplotlib.plot_optimization_history(study)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f'{save_dir}/optimization_history_method_{method}_target_feature_{target_feature}.png')
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f'{save_dir}/slice_method_{method}_target_feature_{target_feature}.png')

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
    file_name = f'{save_dir}/optuna_study_results_method_{method}_target_feature_{target_feature}.json' if n_langs == -1 else f'{save_dir}/optuna_study_results_method_{method}_target_feature_{target_feature}_{n_langs}.json'
    with open(file_name, 'w') as f:
        f.write(results_json)

    best_trial = study.best_trial
    print(f'Best trial: {best_trial}')
    seconds = time.time() - start_time
    print('seconds: {:.2f}'.format(seconds))
    return best_trial.values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for different classifiers.')
    parser.add_argument('--method', type=str, required=True, choices=['svm', 'rf', 'logres', 'xgb', 'dt'],
                        help='The classification method to use.')
    parser.add_argument('--n_langs', type=int, required=True, help='Number of languages to extract data from them and create dataset. Use -1 for using all langs.')
    parser.add_argument('--n_trials', type=int, required=True, help='Number of trials for Optuna.')
    parser.add_argument('--use_filtered', action='store_true', help='Use filtered data if set; otherwise, use unfiltered data.')
    
    args = parser.parse_args()

    # Load data based on whether filtering is enabled
    miltale_X, max_feats = load_data(args.use_filtered)
    target_features = myutils.target_features[100:]

    for target_feature in tqdm(target_features):
        save_dir = 'result' if args.use_filtered else 'result/unfiltered'
        hyperparameter_tuning(args.method, args.n_trials, args.n_langs ,target_feature, max_feats, save_dir=save_dir, use_filtered=args.use_filtered)
