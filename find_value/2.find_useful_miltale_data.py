# This script is used to find out if using miltale data and filtering it is useful for finding the best hyperparameters for the classifiers

import sys
import os
sys.path.append(os.getcwd())
import json
import pandas as pd
import scripts.myutils as myutils


method = 'rf'
save_dir = 'result'


result = {
    'target_feature': list(),
    'filtered_with_miltale_data': list(),
    'filtered_without_miltale_data': list(),
    'unfiltered_with_miltale_data': list(),
    'unfiltered_without_miltale_data': list(),
}
for target_feature in myutils.target_features:
    file_path = f'{save_dir}/unfiltered/optuna_study_results_method_{method}_target_feature_{target_feature}.json'
    if not os.path.isfile(file_path):
        continue
    with open(file_path) as f1:
        unfiltered_d = json.load(f1)
        unfiltered_d_with_miltale_data = [i for i in unfiltered_d if i['params']['miltale_data']]
        unfiltered_d_without_miltale_data = [i for i in unfiltered_d if not i['params']['miltale_data']]
        assert len(unfiltered_d_with_miltale_data) + len(unfiltered_d_without_miltale_data) == len(unfiltered_d)
        unfiltered_best_trial = max(unfiltered_d, key=lambda x:x['value'])
        if unfiltered_best_trial['params']['miltale_data']:
            unfiltered_best_trial_with_miltale_data = max(unfiltered_d_with_miltale_data, key=lambda x:x['value'])
            unfiltered_best_trial_without_miltale_data = max(unfiltered_d_without_miltale_data, key=lambda x:x['value'])

            assert unfiltered_best_trial_with_miltale_data['value'] == unfiltered_best_trial['value']
            assert unfiltered_best_trial_without_miltale_data['value'] <= unfiltered_best_trial['value']
            
            with open(f'{save_dir}/optuna_study_results_method_{method}_target_feature_{target_feature}.json') as f:
                d = json.load(f)
                d_with_miltale_data = [i for i in d if i['params']['miltale_data']]
                d_without_miltale_data = [i for i in d if not i['params']['miltale_data']]
                assert len(d_with_miltale_data) + len(d_without_miltale_data) == len(d)
                best_trial = max(d, key=lambda x:x['value'])
                if best_trial['params']['miltale_data']:
                    best_trial_with_miltale_data = max(d_with_miltale_data, key=lambda x:x['value'])
                    best_trial_without_miltale_data = max(d_without_miltale_data, key=lambda x:x['value'])

                    assert best_trial_with_miltale_data['value'] == best_trial['value']
                    assert best_trial_without_miltale_data['value'] <= best_trial['value']

                    result['target_feature'].append(target_feature)
                    result['filtered_with_miltale_data'].append(best_trial_with_miltale_data['value'])
                    result['filtered_without_miltale_data'].append(best_trial_without_miltale_data['value'])
                    result['unfiltered_with_miltale_data'].append(unfiltered_best_trial_with_miltale_data['value'])
                    result['unfiltered_without_miltale_data'].append(unfiltered_best_trial_without_miltale_data['value'])

result = pd.DataFrame(result)
result.to_csv(f'{save_dir}/useful_miltale.csv')
