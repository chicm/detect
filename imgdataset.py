import os, cv2, glob
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_classes, get_train_ids, get_val_ids, build_bbox_dict

IMG_DIR = r'G:\open-images\224'

class ImageDataset(data.Dataset):
    def __init__(self, img_ids, bbox_dict, has_label = True, transform = None):
        self.img_ids = img_ids
        self.num = len(img_ids)
        self.bbox_dict = bbox_dict
        self.has_label = has_label
        self.transform = transform

    def __getitem__(self, index):
        fn = os.path.join(IMG_DIR, '{}.jpg'.format(self.img_ids[index]))
        img = cv2.imread(fn)
        if self.transforms is not None:
            img = self.transform(img)
        if self.has_label:
            label = self.bbox_dict[self.img_ids[index]]
            return img, label
        else:
            return img

    def __len__(self):
        return self.num

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageDataLoader(object):
    def __init__(self, img_ids, shuffle=True, batch_size=64, transform=None):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.img_ids = img_ids
        self.transform = transform
        self.len = len(img_ids)

        if shuffle:
            self.sampler = torch.utils.data.sampler.RandomSampler(self.img_ids) 
            drop_last = True
        else:
            self.sampler = torch.utils.data.sampler.SequentialSampler(self.img_ids)
            drop_last = False

        self.batch_sampler = torch.utils.data.sampler.BatchSampler(self.sampler, batch_size, drop_last)

        self.sample_iter = iter(self.batch_sampler)

    def __iter__(self):
        return self
    
    def __next__(self):
        indices = next(self.sample_iter)
        ids = [self.img_ids[i] for i in indices]
        fns = [os.path.join(IMG_DIR, '{}.jpg'.format(i)) for i in ids]
        imgs = np.array([cv2.imread(fn) for fn in fns]) / 255.
        imgs = torch.FloatTensor(imgs).cuda()
        return ids, imgs

def get_train_loader(batch_size=64, shuffle = True):
    img_ids = get_train_ids()
    bbox_dict = build_bbox_dict()
    dset = ImageDataset(img_ids, bbox_dict, True, data_transforms)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader

def test_loader():
    loader = ImageDataLoader(get_train_ids())
    for i, data in enumerate(loader):
        if i > 10:
            break
        ids, imgs = data
        print(ids, imgs.size())

if __name__ == '__main__':
    test_loader()