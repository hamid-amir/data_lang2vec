# this script takes around 3 hours to run!
# so I suggest to use the saved matrix instead of running this script.

import pickle
import lang2vec.lang2vec as l2v
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))

# create matrix of phylogency feature(3719 dim) of lang2vec for all 1720 languages
phyl_matrix = np.zeros((len(langs),len(l2v.get_features('eng', 'fam', header=True)['CODE'])))
for i, lang in tqdm(enumerate(langs)):
    phyl_matrix[i] = l2v.get_features(lang, 'fam', header=True)[lang]


# convert the matrix to a sparse one cause phyl_matrix is highly sparse
phyl_matrix_sparse = csr_matrix(phyl_matrix)

# save the resulting sparse matrix for the ease of use  
with open('phyl-matrix-sparse.pickle', 'wb') as f:
    pickle.dump(phyl_matrix_sparse, f)
