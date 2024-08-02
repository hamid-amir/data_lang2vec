import pickle
import time
import sys
import os
import json
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


miltale_langs, miltale_X_sparse = pickle.load(open('miltale_extracted_feats.pickle', 'rb'))
miltale_X = miltale_X_sparse.toarray()
max_feats = min(miltale_X.shape[0], miltale_X.shape[1])


def objective(trial, method: str, target_feature: str):
    features = ['lang_id', 'feat_id', 'geo_lat', 'geo_long', 'lang_group', 'aes_status', 'wiki_size', 'num_speakers', 'lang_fam', 'scripts', 'feat_name', 'phylogency']
    remove_features = [trial.suggest_categorical(feature, [True, False]) for feature in features]
    remove_features = [feature for feature, selected in zip(features, remove_features) if selected]
    miltale_data = trial.suggest_categorical('miltale_data', [True, False])
    if len(remove_features) == len(features):
        remove_features = []

    print(remove_features)
    n_components = trial.suggest_int('n_components', 10, 100)
    miltate_n_components = trial.suggest_int('miltate_n_components', 10, max_feats)
    myutils.extract_features(
        'find_value',
        n_components=n_components,
        miltate_n_components=miltate_n_components,
        # n=300,
        remove_features=remove_features,
        miltale_data=miltale_data,
        job_number=5
    )

    X, Y, names, all_feat_names = pickle.load(open('feats-full_find_value_mil_5.pickle', 'rb'))

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

    


def hyperparameter_tuning(method: str, n_trails: int, target_feature: str, save_dir: str = 'result'):
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, method, target_feature), n_trials=n_trails)

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
    with open(f'{save_dir}/optuna_study_results_method_{method}_target_feature_{target_feature}.json', 'w') as f:
        f.write(results_json)

    best_trial = study.best_trial
    print(f'Best trial: {best_trial}')
    seconds = time.time() - start_time
    print('seconds: {:.2f}'.format(seconds))
    return best_trial.values


if __name__ == '__main__':
    method = sys.argv[1]
    n_trials = int(sys.argv[2])

    # target_features = [
    #     'S_DEMONSTRATIVE_WORD_BEFORE_NOUN', 'S_INDEFINITE_AFFIX', 'S_SEX_MARK', 'S_VOS', 'S_OBJECT_BEFORE_VERB',
    #     'S_OBJECT_AFTER_VERB', 'S_NEGATIVE_WORD_AFTER_OBJECT', 'S_NUMCLASS_MARK', 'P_UVULAR_STOPS', 'P_CLICKS',
    #     'S_CASE_SUFFIX', 'S_XVO', 'S_NEGATIVE_WORD_BEFORE_SUBJECT', 'P_UVULAR_CONTINUANTS', 'S_POLARQ_AFFIX',
    #     'S_CASE_PROCLITIC', 'P_IMPLOSIVES', 'S_POLARQ_WORD', 'S_OBLIQUE_AFTER_VERB', 'S_DEFINITE_AFFIX',
    #     'S_INDEFINITE_WORD', 'P_VELAR_NASAL', 'P_NASAL_VOWELS', 'S_RELATIVE_AROUND_NOUN', 'P_BILABIALS',
    #     'S_POSSESSIVE_HEADMARK', 'S_SUBJECT_BEFORE_OBJECT', 'S_TEND_SUFFIX', 'P_TONE', 'S_OSV', 'P_VELAR_NASAL_INITIAL',
    #     'S_GENDER_MARK', 'P_FRICATIVES', 'P_VOICED_PLOSIVES', 'S_DEMONSTRATIVE_PREFIX', 'S_SUBORDINATOR_WORD_BEFORE_CLAUSE',
    #     'P_COMPLEX_CODAS', 'S_CASE_PREFIX', 'P_EJECTIVES', 'P_CODAS', 'S_OBLIQUE_BEFORE_VERB', 'S_OBJECT_HEADMARK',
    #     'S_ADJECTIVE_AFTER_NOUN', 'S_DEMONSTRATIVE_SUFFIX', 'S_NEGATIVE_PREFIX', 'S_CASE_MARK', 'S_COMITATIVE_VS_INSTRUMENTAL_MARK',
    #     'S_NEGATIVE_WORD_AFTER_VERB', 'S_SOV', 'S_NEGATIVE_WORD_ADJACENT_AFTER_VERB', 'S_NUMERAL_AFTER_NOUN', 'S_ADJECTIVE_WITHOUT_NOUN',
    #     'S_PLURAL_PREFIX', 'S_DEGREE_WORD_AFTER_ADJECTIVE', 'S_POSSESSIVE_DEPMARK', 'S_ADPOSITION_AFTER_NOUN', 'S_SUBJECT_BEFORE_VERB',
    #     'S_PLURAL_WORD', 'P_NASALS', 'S_TEND_PREFIX', 'S_SUBORDINATOR_WORD_AFTER_CLAUSE', 'S_OVS', 'S_NOMINATIVE_VS_ACCUSATIVE_MARK',
    #     'S_DEGREE_WORD_BEFORE_ADJECTIVE', 'S_RELATIVE_AFTER_NOUN', 'S_SVO', 'S_OXV', 'P_LATERAL_L', 'P_LABIAL_VELARS', 'S_DEMONSTRATIVE_WORD_AFTER_NOUN',
    #     'P_LATERAL_OBSTRUENTS', 'S_VSO', 'S_NEGATIVE_WORD_ADJACENT_BEFORE_VERB', 'S_POSSESSOR_AFTER_NOUN', 'P_PHARYNGEALS', 'P_VOICED_FRICATIVES',
    #     'S_OBLIQUE_BEFORE_OBJECT', 'S_POLARQ_MARK_FINAL', 'S_PLURAL_SUFFIX', 'S_NEGATIVE_WORD_AFTER_SUBJECT', 'S_SUBJECT_AFTER_VERB',
    #     'S_PERFECTIVE_VS_IMPERFECTIVE_MARK', 'S_NEGATIVE_WORD_FINAL', 'S_TAM_PREFIX', 'S_VOX', 'S_POSSESSIVE_PREFIX', 'P_GLOTTALIZED_RESONANTS',
    #     'S_PROSUBJECT_CLITIC', 'P_TH', 'S_NEGATIVE_SUFFIX', 'S_DEFINITE_WORD', 'S_FUTURE_AFFIX', 'S_RELATIVE_BEFORE_NOUN', 'S_XOV',
    #     'S_ERGATIVE_VS_ABSOLUTIVE_MARK', 'S_NEGATIVE_AFFIX', 'P_UVULARS', 'S_NUMERAL_BEFORE_NOUN', 'S_NEGATIVE_WORD_BEFORE_VERB', 'S_OVX',
    #     'S_CASE_ENCLITIC', 'S_POLARQ_MARK_SECOND', 'S_ADJECTIVE_BEFORE_NOUN', 'S_PROSUBJECT_WORD', 'S_ADPOSITION_BEFORE_NOUN',
    #     'S_POLARQ_MARK_INITIAL', 'S_NEGATIVE_WORD_INITIAL', 'S_TEND_DEPMARK', 'P_COMPLEX_ONSETS', 'S_POSSESSOR_BEFORE_NOUN',
    #     'P_FRONT_ROUND_VOWELS', 'P_LATERALS', 'S_PAST_VS_PRESENT_MARK', 'S_SUBJECT_AFTER_OBJECT', 'S_TEND_HEADMARK', 'S_OBJECT_DEPMARK',
    #     'S_NEGATIVE_WORD', 'S_TAM_SUFFIX', 'S_NEGATIVE_WORD_BEFORE_OBJECT', 'S_ANY_REDUP', 'S_PROSUBJECT_AFFIX', 'S_OBLIQUE_AFTER_OBJECT',
    #     'S_POSSESSIVE_SUFFIX', 'P_VOICE', 'S_SUBORDINATOR_SUFFIX',
    # ]

    target_features = myutils.target_features[100:]

    for target_feature in tqdm(target_features):
        hyperparameter_tuning(method, n_trials, target_feature)
