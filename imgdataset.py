import os, cv2, glob
import numpy as np

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_classes, get_train_ids, get_val_ids, load_bbox_dict
import settings

IMG_DIR = settings.IMG_DIR

class ImageDataset(data.Dataset):
    def __init__(self, img_ids, bbox_dict, has_label=True):
        self.img_ids = img_ids
        self.num = len(img_ids)
        self.bbox_dict = bbox_dict
        self.has_label = has_label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        fn = os.path.join(IMG_DIR, '{}.jpg'.format(self.img_ids[index]))
        img = cv2.imread(fn)
        img = self.transform(img)
        
        return img, self.img_ids[index]

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
        inputs = torch.stack(imgs)
        
        clfs = []
        bboxes = []
        for _, img_id in batch:
            if not img_id in self.bbox_dict:
                continue
            img_labels = self.bbox_dict[img_id]  # [(cls, [x1, y1, x2, y2]), (...)...]
            clf = [x[0] for x in img_labels]
            clfs.append(clf)
            bbox = [x[1] for x in img_labels]
            bboxes.append(bbox)

        return inputs.cuda(), clfs, bboxes
        '''
        boxes = [x[1][0] for x in batch]
        labels = [x[1][1] for x in batch]

        inputs = torch.stack(imgs)
        input_size = torch.Tensor(list(inputs.size()[-2:]))
        bbox_targets, clf_targets = [], []
        for box, label in zip(boxes, labels):
            bbox_target, clf_target = self.target_encoder.encode(box, label, input_size=input_size)
            bbox_targets.append(bbox_target)
            clf_targets.append(clf_target)

        bbox_targets, clf_targets = torch.stack(bbox_targets), torch.stack(clf_targets)
        clf_targets = clf_targets.unsqueeze(-1)
        targets = torch.cat((bbox_targets, clf_targets), 2)
        return inputs, targets
        '''

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
    bbox_dict = load_bbox_dict()
    dset = ImageDataset(img_ids, bbox_dict, True)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def test_loader():
    #loader = ImageDataLoader(get_train_ids())
    loader = get_train_loader()
    for i, data in enumerate(loader):
        #if i > 10:
        #    break
        imgs, clfs, bbox = data
        print(clfs, bbox, imgs.size())

if __name__ == '__main__':
    test_loader()