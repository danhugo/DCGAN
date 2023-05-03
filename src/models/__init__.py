# from src import BASE_DIR
import os
from src import BASE_DIR

LSUN_MODEL_DIR = os.path.join(BASE_DIR,'models/LSUN')
LSUN_REPORT_DIR = os.path.join(BASE_DIR,'reports/LSUN')
LSUN_REPORT_IMAGE_DIR = os.path.join(BASE_DIR, 'reports/LSUN/images')
CONFIG_PATH = os.path.join(BASE_DIR, 'src/cfg/LSUN_dcgan.yaml')

if not os.path.exists(LSUN_REPORT_DIR):
    os.mkdir(LSUN_REPORT_DIR)
if not os.path.exists(LSUN_REPORT_IMAGE_DIR):
    os.mkdir(LSUN_REPORT_IMAGE_DIR)
if not os.path.exists(LSUN_MODEL_DIR):
    os.mkdir(LSUN_MODEL_DIR)

