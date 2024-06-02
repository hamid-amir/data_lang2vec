import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils


myutils.extract_features(classifier='find_value', n_components=10)