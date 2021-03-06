import os, cv2, glob
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_classes, build_bbox_dict, get_class_id_converters, \
    build_bbox_dict, build_small_bbox_dict, load_small_train_ids, get_val_ids
import settings

def check_classes():
    tmp = get_classes()
    print(tmp[:10])
    classes = set(tmp)
    print(len(classes))
    bbox = build_bbox_dict()
    v = bbox.values()
    print(len(v))
    for x in v:
        for c, _ in x:
            if not (c in classes):
                print(c)
    print('done')

def check_test_ids():
    test_img_ids = [os.path.basename(x).split('.')[0] for x in  glob.glob(os.path.join(settings.TEST_IMG_DIR, '*.jpg'))]
    print(len(test_img_ids))
    print(test_img_ids[:10])

    df = pd.read_csv(settings.SAMPLE_SUB_FILE)
    print(df.head())
    submit_ids = df.values[:, 0].tolist()
    print(submit_ids[:10])
    print(len(submit_ids))
    assert(set(test_img_ids) == set(submit_ids))


def build_dict():
    #itos, stoi = get_class_id_converters()
    #build_bbox_dict(stoi)
    #test_bboxes()
    #test_show()
    #test_class_converters()
    build_small_bbox_dict()
    d, imgids = load_small_train_ids()
    print(imgids[:10])
    for imgid in imgids[:10]:
        print(d[imgid])


if __name__ == '__main__':
    #check_classes()
    #check_test_ids()
    #print(get_val_ids()[:10])
    build_dict()