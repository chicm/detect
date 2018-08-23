import os, cv2, glob
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_classes, build_bbox_dict


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

if __name__ == '__main__':
    check_classes()