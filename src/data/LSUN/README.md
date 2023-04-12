# LSUN
This paper used bedroom dataset in LSUN dataset to train the model.  
To exprerience with LSUN dataset, you need to have these libs in your environment.
- cURL
- numpy
- lmdb
- opencv

The paper performed a simple image de-duplication manipulation on dataset by using a simple Autoencoder to convert images to lower representation codes and comparing distances among these codes to extract the redundant images. This method is not performed in this code.  

Additionally, Images are scaled down to 32x32 resolution and cropped to square centrally. 

## Usage
Download functional files. 

```
wget https://raw.githubusercontent.com/fyu/lsun/master/data.py
wget https://raw.githubusercontent.com/fyu/lsun/master/download.py

```

After downloading download.py file. Run download data command.
```
python download.py -c bedroom
```
Two zip files: **bedrrom_train_lmdb.zip** and **bedroom_val_lmdb.zip** have been downloaded. In current folder, unzip two files.
```
unzip bedroom_train_lmdb.zip
unzip bedroom_val_lmdb.zip
```
After unzip these file, we can follow instruction from [here](https://github.com/fyu/lsun) to view and export the images. When extracting the images, should use **--flat** to import all images directly to the destination folder.

## Processing
The paper downsamples images to 32x32. In this code I downscale it to 64x64.