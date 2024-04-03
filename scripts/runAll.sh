# First, download the raw data
./scripts/0.getData.sh

# Get the lang2vec vectors
python3 scripts/1.lang2vec_data.py

# Find the values that are likely to be missing
python3 scripts/2.find_missing_vals.py
