import os
import glob
import collections
import cv2
import numpy as np
import pandas as pd

DATA_DIR = '/media/chicm/NVME/open-images'
IMG_DIR = '/media/chicm/NVME/open-images'
MC_CSV = 'mc.csv'
MBB_CSV = 'mbb.csv'

def build_bbox_dict():
    bbox_file = 'challenge-2018-train-annotations-bbox.csv'
    bbox_dict = collections.defaultdict(lambda: [])
    with open(os.path.join(DATA_DIR, bbox_file), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            row = line.strip().split(',')
            bbox_dict[row[0]].append((row[2], [float(row[4]), float(row[6]), float(row[5]), float(row[7])]))
    return bbox_dict

def draw_img(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def draw_screen_rect(image, bbox, color, alpha=0.5):
    H, W = image.shape[:2]
    x1, y1 = round(bbox[0]*W), round(bbox[1]*H)
    x2, y2 = round(bbox[2]*W), round(bbox[3]*H)
    #image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def build_csvs_from_subset_dir(subset_path):
    bbox_dict = build_bbox_dict()
    filenames = glob.glob(os.path.join(IMG_DIR, subset_path, '*.jpg'))
    print(len(filenames))
    fns = [os.path.basename(o) for o in filenames]
    mcs = [' '.join([str(o[0]) for o in bbox_dict[fn.split('.')[0]]]) for fn in fns]
    df1 = pd.DataFrame({'fn': fns, 'clas': mcs}, columns=['fn', 'clas'])
    df1.to_csv(MC_CSV, index=False)

    mbb = [' '.join([' '.join([str(i) for i in o[1]]) for o in bbox_dict[fn.split('.')[0]]]) for fn in fns]
    df2 = pd.DataFrame({'fn': fns, 'bbox': mbb}, columns=['fn','bbox'])
    df2.to_csv(MBB_CSV, index=False)

def get_fn(img_id):
    return os.path.join(IMG_DIR, 'train_0', '{}.jpg'.format(img_id))

def show_img_with_label(img_id):
    bbox_dict = build_bbox_dict()
    bb = bbox_dict[img_id]
    print(bb)
    img = cv2.imread(get_fn(img_id))
    for b in bb:
        draw_screen_rect(img, b[1], [0,0,255])
    draw_img(img_id, img)
    cv2.waitKey(0)

def show_img_with_label2(img_id):
    print(img_id)
    df = pd.read_csv(MBB_CSV)
    bb = df[df['fn'] == img_id+'.jpg']['bbox'].values[0].split(' ')

    print(bb)
    img = cv2.imread(get_fn(img_id))
    b = []
    for i, o in enumerate(bb):
        print(o)
        b.append(float(o))
        if (i +1) % 4 == 0: 
            draw_screen_rect(img, b, [0,255,0])
            b = []
    draw_img(img_id, img)
    cv2.waitKey(0)

def test_show():
    #build_csvs_from_subset_dir('train_0')
    filenames = glob.glob(os.path.join(IMG_DIR, 'train_0', '*.jpg'))
    img_id = os.path.basename(filenames[5]).split('.')[0]
    show_img_with_label2(img_id)
    show_img_with_label(img_id)

import sys
if __name__ == '__main__':
    build_csvs_from_subset_dir('train_0')
    sys.path.insert(0,'/home/chicm/ml/fastai')
    from fastai.dataset import *
    from fastai.conv_learner import *

    sz = 224
    f_model=resnet34
    tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO)
    md = ImageClassifierData.from_csv(IMG_DIR, 'train_0', MC_CSV, tfms=tfms, bs=64)
    learn = ConvLearner.pretrained(f_model, md)
    learn.opt_fn = optim.Adam
    lrf=learn.lr_find(1e-5,100)
    lr = 2e-2
    learn.fit(lr, 1, cycle_len=3, use_clr=(32,5))
    
    print('done')

