import pickle
import json
import lang2vec.lang2vec as l2v
import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils
from sklearn.model_selection import cross_val_predict
from find_missing.utils import cross_validation



def main(save_dir: str = 'result'):
    methods = ['svm', 'rf', 'logres', 'knn', 'gbc', 'dt']
    best_value, best_method = 0, 'svm'
    for method in methods:
        file_path = f'{save_dir}/optuna_study_results_{method}.json'
        if os.path.exists(file_path):
            with open(file_path) as f:
                d = json.load(f)
            d_sorted = sorted(d, key=lambda x: x['value'], reverse=True)
            best = d_sorted[0]
            if best['value'] > best_value:
                best_all = best
                best_value = best['value']
                best_method = method

    for method in methods:
        file_path = f'{save_dir}/optuna_study_results_{method}.json'
        if os.path.exists(file_path):
            with open(file_path) as f:
                d = json.load(f)
            d_sorted = sorted(d, key=lambda x: x['value'], reverse=True)
            best = d_sorted[0]

            print(f"{method} f1 score after optimizing hyperparameters on 300 langs: {best['value']}")
            print('Training it with the optimized hyperparameters on all langs...')

            remove_features = []
            for param in best['params']:
                if param in ['lang_id', 'feat_id', 'geo_lat', 'geo_long', 'lang_group', 'aes_status', 'wiki_size', 'num_speakers', 'lang_fam', 'scripts', 'feat_name', 'phylogency']:
                    if best['params'][param]:
                        remove_features.append(param)


            # construct dataset using all langs, based on the best settings for the classifier 
            myutils.extract_features(
                'find_missing',
                n_components=best['params']['n_components'],
                remove_features=remove_features,
            )
            x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))


            if method == 'svm':
                from sklearn import svm
                clf = svm.SVC(C=best['params']['C'], 
                            kernel=best['params']['kernel'], 
                            degree=best['params']['degree'], 
                            gamma=best['params']['gamma'], 
                            random_state=myutils.seed, 
                            probability=True)
                if best_method == 'svm': best_clf = clf
                print(f"Training best SVM on all langs using C={best['params']['C']}, "
                      f"kernel={best['params']['kernel']}, degree={best['params']['degree']}, gamma={best['params']['gamma']}")

            elif method == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(
                    max_depth=best['params']['max_depth'], 
                    min_samples_split=best['params']['min_samples_split'],
                    criterion=best['params']['criterion'], 
                    class_weight=best['params']['class_weight'] if best['params']['class_weight'] != 'None' else None,
                    n_estimators=best['params']['n_estimators'], 
                    random_state=myutils.seed, 
                    n_jobs=28
                )
                if best_method == 'rf': best_clf = clf
                print(f"Training best Random Forest on all langs using max_depth={best['params']['max_depth']}, "
                    f"min_samples_split={best['params']['min_samples_split']}, criterion={best['params']['criterion']}, "
                    f"class_weight={best['params']['class_weight']}, n_estimators={best['params']['n_estimators']}")

            elif method == 'logres':
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(penalty=best['params']['penalty'], 
                                        tol=best['params']['tol'], 
                                        C=best['params']['C'], 
                                        solver=best['params']['solver'], 
                                        random_state=myutils.seed, 
                                        n_jobs=28, 
                                        max_iter=500)
                if best_method == 'logres': best_clf = clf
                print(f"Training best find_missing classifier which is a Logistic Regression with penalty={best['params']['penalty']}, "
                    f"tol={best['params']['tol']}, C={best['params']['C']}, solver={best['params']['solver']}")


            # PART 1: Report the best f1 score of each classifier
            result = cross_validation(clf, x, y)
            print(f"{method} f1 score after optimizing hyperparameters on all langs: {result['f1']}")



    # PART2: Make predictions based on the best classifier among all of them to extract most probable missing features
    print(f'-----> Using the best classifier which is {best_method} to do the predictions and extract most probable missing features...')
    # K-fold Cross-validation and prediction: k=5
    pred_y = cross_val_predict(best_clf, x, y, method='predict_proba', cv=5).tolist()

    # Collect datapoints with gold data but likely missing
    data_points = []
    for i in range(len(pred_y)):
        if y[i] == 1:
            instance = [pred_y[i][0], names[i]]  # [missing_prob, lang|feat]
            data_points.append(instance)

    l2v_langs, l2v_data, _ = pickle.load(open('lang2vec.pickle', 'rb'))
    feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']

    # Add the gold value from the lang2vec database
    final_data = []
    for item in sorted(data_points):
        lang, feat_name = item[1].split('|')
        feat_idx = feature_names.index(feat_name)
        lang_idx = l2v_langs.index(lang)
        gold_val = l2v_data[lang_idx][feat_idx]
        final_data.append([lang, feat_name, gold_val])

    # print(len(final_data))

    # Save the sorted data
    pickle.dump(final_data, open('final_data.pickle', 'wb'))




if __name__ == '__main__':
    main()