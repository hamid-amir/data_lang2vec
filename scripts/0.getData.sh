# note that this takes a long time to download (5-7 days)
cd data
echo "get LTI-LangID-rel5.txz from https://sourceforge.net/projects/la-strings/files/Language-Data/LTI-LangID-rel5.txz/download"
tar Jxvf LTI-LangID-rel5.txz
cd MIL-TALE/5
cp ../../scripts/LTI-LangID-rel5-errata.txz .
tar Jxvf LTI-LangID-rel5-errata.tx
./code/install.sh
# ../../MILTALE
# Include downloadable languages? (y/N) y
#    Retrieve downloadable languages? (takes 2-3 days) (y/N) y
# Include 'Additional' languages? (y/N) y
# Copy 'Sample' languages? (y/N) n
# Merge training data into single file per language+script pair? (y/N) n
# Compress training sets? (y/N) n
# Generate reduced-size training sets? (y/N) n
# Generate devtest-only training set(s)? (y/N) n
cd ../../
python3 scripts/0.merge_miltale.py > merge.sh
chmod +x merge.sh
./merge.sh

