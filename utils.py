import os
import glob
import collections
import cv2
import numpy as np

DATA_DIR = '/media/chicm/NVME/open-images'
IMG_DIR = '/media/chicm/NVME/open-images/train_0'

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
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 4)

def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)

if __name__ == '__main__':
    bbox_dict = build_bbox_dict()
    bb = bbox_dict['06b14f282ecf3aec']
    print(bb)
    img = cv2.imread(os.path.join(IMG_DIR, '06b14f282ecf3aec.jpg'))
    for b in bb:
        draw_screen_rect(img, b[1], [0,255,255])
    draw_img('06b14f282ecf3aec.jpg', img)
    cv2.waitKey(0)