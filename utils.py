import os
import glob
import collections
import cv2
import numpy as np
import pandas as pd
import pickle
import time
import settings

IMG_DIR = settings.IMG_DIR
VAL_FILE = settings.VAL_FILE
CLASS_FILE = settings.CLASS_FILE
BBOX_FILE = settings.BBOX_FILE
BBOX_BIN_FILE = os.path.join(settings.DATA_DIR, 'bbox.pk')

BAD_IMG_IDS = set([])

MC_CSV = 'mc.csv'
MBB_CSV = 'mbb.csv'

def get_classes():
    classes = []
    with open(CLASS_FILE, 'r') as f:
        for line in f:
            classes.append(line.strip().split(',')[0])
    return classes

def get_class_dict():
    class_dict = {}
    with open(CLASS_FILE, 'r') as f:
        for line in f:
            print(line)
            k, v = line.strip().split(',')
            class_dict[k] = v
    return class_dict

def get_class_id_converters():
    itos = get_classes()
    stoi = {itos[i]: i for i in range(len(itos))}
    return itos, stoi

def get_val_ids():
    val_ids = []
    with open(VAL_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            val_ids.append(line.strip())
    return val_ids


def get_train_ids():
    filenames = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
    #print(len(filenames))
    img_ids = [os.path.basename(fn).split('.')[0] for fn in filenames]
    valset = set(get_val_ids())
    img_ids = [img_id for img_id in img_ids if not (img_id in valset or img_id in BAD_IMG_IDS)]
    #print(len(img_ids))
    return img_ids

def build_bbox_dict(cls_stoi):
    bbox_dict = {} #collections.defaultdict(lambda: [])
    with open(BBOX_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            row = line.strip().split(',')
            if row[0] in bbox_dict:
                # return (class, [x1, y1, x2, y2])
                bbox_dict[row[0]].append((cls_stoi[row[2]], [float(row[4]), float(row[6]), float(row[5]), float(row[7])]))
            else:
                bbox_dict[row[0]] = []
    with open(BBOX_BIN_FILE, 'wb') as f:
        pickle.dump(bbox_dict, f)
    return bbox_dict

def load_bbox_dict():
    with open(BBOX_BIN_FILE, 'rb') as f:
        return pickle.load(f)

def draw_img(image, name = '', resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))

def draw_screen_rect(image, bbox, color=[0,0,255], alpha=0.5):
    H, W = image.shape[:2]
    x1, y1 = round(bbox[0]*W), round(bbox[1]*H)
    x2, y2 = round(bbox[2]*W), round(bbox[3]*H)
    #image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

def draw_shadow_text(img, text, pt, color=(255, 0, 0), fontScale=0.5, thickness=1):
    #if color1 is None: color1=(0,0,0)
    #if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color, thickness, cv2.LINE_AA)
    #cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

def build_csvs_from_subset_dir(subset_path):
    bbox_dict = build_bbox_dict()
    filenames = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
    print(len(filenames))
    fns = [os.path.basename(o) for o in filenames]
    mcs = [' '.join([str(o[0]) for o in bbox_dict[fn.split('.')[0]]]) for fn in fns]
    df1 = pd.DataFrame({'fn': fns, 'clas': mcs}, columns=['fn', 'clas'])
    df1.to_csv(MC_CSV, index=False)

    mbb = [' '.join([' '.join([str(i) for i in o[1]]) for o in bbox_dict[fn.split('.')[0]]]) for fn in fns]
    df2 = pd.DataFrame({'fn': fns, 'bbox': mbb}, columns=['fn','bbox'])
    df2.to_csv(MBB_CSV, index=False)

def get_fn(img_id):
    return os.path.join(IMG_DIR, '{}.jpg'.format(img_id))

def show_img_with_label(img, bb):
    itos, stoi = get_class_id_converters()
    class_dict = get_class_dict()
    #bb = bbox_dict[img_id]
    #print(img_id)
    print(bb)
    #img = cv2.imread(get_fn(img_id))
    for b in bb:
        draw_screen_rect(img, b[1])
        text_x = round(b[1][0] * img.shape[0])
        text_y = round(b[1][1] * img.shape[1]) + 10
        cls_name = class_dict[itos[b[0]]]
        draw_shadow_text(img, cls_name, (text_x, text_y))
    draw_img(img)
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
    #filenames = glob.glob(IMG_DIR, '*.jpg'))
    #img_ids = os.path.basename(filenames[5]).split('.')[0]
    #show_img_with_label2(img_id)
    img_id = '6f4e88d1573905ac'
    img = cv2.imread(get_fn(img_id))
    bbox_dict = load_bbox_dict()
    bbox = bbox_dict[img_id] #[(10, [0.0, 0.086875, 0.713884, 0.52375]), (29, [0.396811, 0.1575, 0.999062, 0.999375]), (251, [0.090056, 0.091875, 0.741088, 0.35625])]
    show_img_with_label(img, bbox)

#import sys
#if __name__ == '__main__':
'''
def test():
    build_csvs_from_subset_dir('train_0')
    sys.path.insert(0,'/home/chicm/ml/fastai')
    #from fastai.dataset import *
    #from fastai.conv_learner import *

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
'''
def test_class_converters():
    itos, stoi = get_class_id_converters()
    indices = [1, 5, 10, 118, 225, 499]
    clas = [itos[i] for i in indices]
    print(clas)
    ind2 = [stoi[clas[i]] for i in range(len(clas))]
    print(ind2)

def test_bboxes():
    bbox_dict = load_bbox_dict()
    print(len(bbox_dict))
    train_ids = get_train_ids()
    print(len(train_ids))
    count = 0
    for id in train_ids:
        if not id in bbox_dict:
            count += 1
    print(count)

if __name__ == '__main__':
    test_bboxes()
    #test_show()
    #test_class_converters()

    #start = time.time()
    #print(len(build_bbox_dict()))
    #print(str(time.time() - start))
    #print(len(load_bbox_dict()))
    #print(str(time.time() - start))

    #_, cls_stoi = get_class_id_converters()
    #build_bbox_dict(cls_stoi)

