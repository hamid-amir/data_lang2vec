# First, download the raw data
./scripts/0.getData.sh

# Get the lang2vec vectors
python3 scripts/1.lang2vec_data.py

# Find the values that are likely to be missing
# Extract features
python3 scripts/2.find_missing.get_feats.py
# Train
python3 scripts/2.find_missing.train.py rf
python3 scripts/2.find_missing.train.py logres
python3 scripts/2.find_missing.train.py svm
