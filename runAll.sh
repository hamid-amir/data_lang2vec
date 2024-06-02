# First, download the raw data
./scripts/0.getData.sh

# Get the lang2vec vectors
python3 scripts/1.lang2vec_data.py


# Find the values that are likely to be missing -> find_missing classifier
# Extract features
python3 find_missing/0.get_feats.py
# Train
python3 find_missing/2.train.py rf
python3 find_missing/2.train.py logres
python3 find_missing/2.train.py svm


# Predict the features values of present ones -> find_value classifier
# Extract features
python3 find_value/0.get_feats.py
# Train
python3 find_value/1.train.py rf
python3 find_value/1.train.py logres
python3 find_value/1.train.py svm