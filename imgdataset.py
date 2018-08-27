import os, cv2, glob
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_classes, get_boxed_train_ids, get_val_ids, load_bbox_dict, load_small_train_ids, get_class_names
from encoder import DataEncoder
import settings

IMG_DIR = settings.IMG_DIR

class ImageDataset(data.Dataset):
    def __init__(self, img_ids, bbox_dict, has_label=True):
        self.input_size = settings.IMG_SZ
        self.img_ids = img_ids
        self.num = len(img_ids)
        self.bbox_dict = bbox_dict
        self.has_label = has_label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        for img_id in self.img_ids:
            box = []
            label = []
            if img_id in self.bbox_dict:
                for x in self.bbox_dict[img_id]:
                    box.append(x[1])
                    label.append(x[0])
            else:
                raise ValueError('No bbox: {}'.format(img_id))
            self.boxes.append(torch.Tensor(box)*self.input_size) # 
            self.labels.append(torch.LongTensor(label)) #



    def __getitem__(self, index):
        fn = os.path.join(IMG_DIR, '{}.jpg'.format(self.img_ids[index]))
        img = cv2.imread(fn)
        img = self.transform(img)
        #print(get_class_names(self.labels[index]))
        
        return img, self.boxes[index], self.labels[index]

    def __len__(self):
        return self.num

    def collate_fn(self, batch):
        """Encode targets.

        Args:
          batch: (list) of images, ids

        Returns:
          images, stacked bbox_targets, stacked clf_targets.
        """
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            #print('1>>>')
            #print(boxes[i].size(), labels[i].size())
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

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

def get_train_loader(img_dir=IMG_DIR, batch_size=8, shuffle = True):
    bbox_dict = load_bbox_dict()
    img_ids = get_boxed_train_ids(bbox_dict, img_dir=img_dir)
    print(len(img_ids))
    print(img_ids[:10])

    dset = ImageDataset(img_ids, bbox_dict, True)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def get_small_train_loader():
    small_dict, img_ids = load_small_train_ids()
    print(img_ids[:10])
    dset = ImageDataset(img_ids[:10], small_dict)
    dloader = data.DataLoader(dset, batch_size=2, shuffle=False, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def test_loader():
    #loader = ImageDataLoader(get_train_ids())
    loader = get_small_train_loader()
    for i, data in enumerate(loader):
        imgs, bbox, clfs = data
        print(imgs.size(), bbox.size(), clfs.size())
        print(torch.max(bbox))

if __name__ == '__main__':
    test_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])