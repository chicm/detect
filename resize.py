import os
import glob
import cv2
import argparse

H, W = 224, 224

def resize(src_dir, tgt_dir):
    filenames = glob.glob(os.path.join(src_dir, '*.jpg'))
    print(len(filenames))
    for i, fn in enumerate(filenames):
        print('{:06d}/{} {}'.format(i, len(filenames), fn), end='\r')
        img = cv2.imread(fn)
        img = cv2.resize(img, (H, W))
        tgt_fn = os.path.join(tgt_dir, os.path.basename(fn))
        cv2.imwrite(tgt_fn, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--tgt_dir', type=str, default=r'd:\kaggle\open-images\224', required=False)
    args = parser.parse_args()

    resize(args.src_dir, args.tgt_dir)
    #resize(r'd:\kaggle\open-images\train_0\*.jpg', )