import sys
import os
sys.path.append(os.getcwd())

import scripts.myutils as myutils

features = [
    # 'lang_id',
    'feat_id',
    'geo_lat',
    'geo_long',
    'lang_group',
    'aes_status',
    'wiki_size',
    'num_speakers',
    'lang_fam',
    'scripts',
    'feat_name',
    # 'phylogency',
    # 'inventory_average',
    'phonology_average',
    # 'geo',
]

myutils.extract_features(
    'find_value',
    n_components=100,
    miltate_n_components=0,
    # n=100,
    remove_features=features,
    miltale_data=False,
    job_number=0,
    use_filtered=False
)