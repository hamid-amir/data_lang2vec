import random
from tqdm import tqdm
import os
import lang2vec.lang2vec as l2v


miltale_langs = set()
for lang_file in os.listdir('data/MILTALE-CLEAN/'):
    lang_code = lang_file.split('.')[0]
    miltale_langs.add(lang_code)

lang2vec_langs = []
for lang in l2v.LANGUAGES:
   lang2vec_langs.append(lang)

common_langs = list(set(miltale_langs) & set(lang2vec_langs))



typo_source = 'syntax_wals'
total, unmissing = 0, 0
all_pres_syntax_wals = {}

for lang in tqdm(common_langs):
    sw_features_names = l2v.get_features(lang, typo_source, header=True)['CODE']
    sw_features_values = l2v.get_features(lang, 'syntax_wals', header=True)[lang]
    for name, value in zip(sw_features_names, sw_features_values):
        if value != '--':
            unmissing += 1
            all_pres_syntax_wals[f'{lang}_{name}'] = value
        total += 1



seed = 42
train_y, test_y, dev_y = {}, {}, {}

for key, value in all_pres_syntax_wals.items():
    rand = random.random()
    if rand < 0.7:
        train_y[key] = value
    elif 0.7 < rand < 0.85:
        test_y[key] = value
    else:
        dev_y[key] = value
