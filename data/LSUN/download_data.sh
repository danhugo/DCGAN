# download files from LSUN github repository to download LSUN data
curl -O https://raw.githubusercontent.com/fyu/lsun/master/data.py
curl -O https://raw.githubusercontent.com/fyu/lsun/master/download.py

echo "Downloading bedroom data ..."
python3 download.py -c bedroom

# extract zip file to path
echo "Extracting bedroom data ..."
unzip bedroom_train_lmdb.zip 
unzip bedroom_val_lmdb.zip
