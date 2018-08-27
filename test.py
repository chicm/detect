import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os
import settings

from utils import load_small_train_ids, get_class_names

small_dict, ids = load_small_train_ids()

test_ids = ['0000048549557964', '0000071d71a0a6f6', '000018acd19b4ad3', '00001bcc92282a38', '0000201cd362f303', '000020780ccee28d', '000023aa04ab09ed', '0000253ea4ecbf19', '0000286a5c6a3eb5', '00002f4ff380c64c']
img_id = test_ids[0]

print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('./model/best_512.pth'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open(os.path.join(settings.IMG_DIR, img_id+'.jpg'))
w = h = settings.IMG_SZ
#img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))

draw = ImageDraw.Draw(img)
print(boxes)
print(get_class_names(list(labels)))
for box in boxes:
    draw.rectangle(list(box), outline='red')

trueboxes = torch.Tensor([b[1] for b in small_dict[img_id]])*settings.IMG_SZ
truelabels = [b[0] for b in small_dict[img_id]]
print(trueboxes)
print(get_class_names(truelabels))
for box in trueboxes:
    draw.rectangle(list(box), outline='blue')
img.show()
