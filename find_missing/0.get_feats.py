import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils


myutils.extract_features(classifier='find_missing', n_components=10)