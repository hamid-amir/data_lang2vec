import sys
import os
sys.path.append(os.getcwd())

import json

import scripts.myutils as myutils

method = 'rf'
save_dir = 'result'
with open(f'result/knn.json') as f:
    knn_results = json.load(f)

rf_values = list()
merge_values = list()
results = list()
for target_feature in myutils.target_features:
    with open(f'{save_dir}/result/optuna_study_results_method_{method}_target_feature_{target_feature}.json') as f:
        d = json.load(f)
        best_trial = max(d, key=lambda x:x['value'])
        rf_values.append(best_trial['value'])
        knn = knn_results[target_feature]
        merge_values.append(max(knn, best_trial['value']))
        best_trial['knn'] = knn
        best_trial['feature_name'] = target_feature
        results.append(best_trial)
        if knn > best_trial['value']:
            print(target_feature)

sorted_results = sorted(results, key=lambda x: x['knn'] - x["value"])
with open(f'result/knn_comparison.json', 'w') as f:
    f.write(json.dumps(sorted_results, indent=4))

print(f'avg knn: {sum(knn_results.values()) / len(knn_results)}')
print(f'avg rf: {sum(rf_values) / len(rf_values)}')
print(f'avg merge: {sum(merge_values) / len(merge_values)}')