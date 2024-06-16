import pickle
import sys
import os

sys.path.append(os.getcwd())
from find_missing.utils import cross_validation, get_clf
from scripts import myutils

myutils.extract_features('find_missing', n_components=0, n=100)
x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))

method = sys.argv[1]
clf = get_clf(method)
cross_validation(clf, x, y)
