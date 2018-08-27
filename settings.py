import os

DATA_DIR = r'D:\data\detect'
IMG_DIR = os.path.join(DATA_DIR, 'train', '512_1')
ANNO_DIR = os.path.join(DATA_DIR, 'annotations')

VAL_FILE = os.path.join(ANNO_DIR, 'challenge-2018-image-ids-valset-od.csv')
CLASS_FILE =  os.path.join(ANNO_DIR, 'challenge-2018-class-descriptions-500.csv')
BBOX_FILE = os.path.join(ANNO_DIR, 'challenge-2018-train-annotations-bbox.csv')