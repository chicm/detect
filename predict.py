import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import time
import os
import settings
import pandas as pd

from utils import load_small_train_ids, get_class_names, get_class_id_converters
from imgdataset import get_test_loader

CKP_FILE = './ckps/best_3.pth'
batch_size = 24

itos, stoi = get_class_id_converters()


def _get_bbox_relative(bbox):
    h = settings.IMG_SZ
    w = settings.IMG_SZ
    x_min = max(0.0, bbox[0] / h)
    y_min = max(0.0, bbox[1] / w)
    x_max = min(1.0, bbox[2] / h)
    y_max = min(1.0, bbox[3] / w)
    result = [x_min, y_min, x_max, y_max]
    return [str(r) for r in result]

def _get_prediction_string(bboxes, labels, scores):
    prediction_list = []
    for bbox, label, score in zip(bboxes, labels, scores):
        prediction_list.append(itos[label])
        prediction_list.append(str(score))
        prediction_list.extend(_get_bbox_relative(bbox))
    prediction_string = " ".join(prediction_list)
    return prediction_string

def transform(image_ids, results):
    #self.decoder_dict = decoder_dict
    prediction_strings = []
    for bboxes, labels, scores in results:
        prediction_strings.append(_get_prediction_string(bboxes, labels, scores))
    submission = pd.DataFrame({'ImageId': image_ids, 'PredictionString': prediction_strings})
    return {'submission': submission}


def predict():
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    print('==> Preparing data..')

    dloader = get_test_loader(img_dir=settings.TEST_IMG_DIR, batch_size=batch_size)
    print(dloader.num)

    # Model
    net = RetinaNet()
    net.load_state_dict(torch.load(CKP_FILE))
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    bgtime = time.time()
    encoder = DataEncoder()
    net.eval()
    prediction_strings = []
    for batch_idx, inputs in enumerate(dloader):
        inputs = Variable(inputs.cuda())
        
        loc_preds, cls_preds = net(inputs)
        print('{} / {}  {:.2f}'.format(batch_size*(batch_idx+1), dloader.num, (time.time() - bgtime)/60), end='\r')
        for i in range(len(loc_preds)):
            boxes, labels, scores = encoder.decode(loc_preds[i].data.cpu(), cls_preds[i].data.cpu(), (settings.IMG_SZ, settings.IMG_SZ))
            prediction_strings.append(_get_prediction_string(boxes, labels, scores))
    print(len(prediction_strings))
    print(prediction_strings[:3])
    submission = pd.DataFrame({'ImageId': dloader.img_ids, 'PredictionString': prediction_strings})
    submission.to_csv('sub1.csv', index=False)
            

if __name__ == '__main__':
    predict()

