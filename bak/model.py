import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

drop = 0.2
ncls = 500

## P layers ## ---------------------------
class LateralBlock(nn.Module):
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top     = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c , p):
        _,_,H,W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2,mode='nearest')
        p = p[:,:,:H,:W] + c
        p = self.top(p)

        return p

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

## C layers ## ---------------------------
class SEBottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1, reduction=16):
        super(SEBottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.bn1   = nn.BatchNorm2d(in_planes,eps = 2e-5)
        self.conv1 = nn.Conv2d(in_planes,     planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv2 = nn.Conv2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv3 = nn.Conv2d(   planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.se = SELayer(planes*4, reduction)
        if is_downsample:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)


    def forward(self, x):

        x = F.relu(self.bn1(x),inplace=True)
        z = self.conv1(x)
        z = F.relu(self.bn2(z),inplace=True)
        z = self.conv2(z)
        z = F.relu(self.bn3(z),inplace=True)
        z = self.conv3(z)
        z = self.se(z)

        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x

        return z



def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)



def make_layer_c(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(SEBottleneckBlock(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SEBottleneckBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)



class FeatureNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=512 ):
        super(FeatureNet, self).__init__()

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)


        self.layer_c1 = make_layer_c(   64,  64,  256, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(  256, 128,  512, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(  512, 256, 1024, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c( 1024, 512, 2048, num_blocks=3, stride=2)  #out = 512*4 = 2048


        # top-down
        self.layer_p4 = nn.Conv2d   ( 2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock( 1024, out_channels, out_channels)
        self.layer_p2 = LateralBlock(  512, out_channels, out_channels)
        self.layer_p1 = LateralBlock(  256, out_channels, out_channels)



    def forward(self, x):
        #pass                        #; print('input ',   x.size())
        c0 = self.layer_c0 (x)       #; print('layer_c0 ',c0.size())
                                     #
        c1 = self.layer_c1(c0)       #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)       #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)       #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)       #; print('layer_c4 ',c4.size())


        p4 = self.layer_p4(c4)       #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)   #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)   #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)   #; print('layer_p1 ',p1.size())

        features = [p1,p2,p3,p4]
        #assert(len(self.cfg.rpn_scales) == len(features))
        print(p1.size(), p2.size(), p3.size(), p4.size())

        return features



class StdConv(nn.Module):
    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x): return self.drop(self.bn(F.relu(self.conv(x))))
        
def flatten_conv(x,k):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)

class OutConv(nn.Module):
    def __init__(self, k, nin, bias):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, (ncls + 1)*k, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, 4*k, 3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self, x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]

class SSD_Head(nn.Module):
    def __init__(self, k, bias):
        super().__init__()
        self.drop = nn.Dropout(0.25)
        self.sconv0 = StdConv(512,256, stride=1)
        self.sconv1 = StdConv(256,256)
        self.sconv2 = StdConv(256,256)
        self.out = OutConv(k, 256, bias)
        
    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        return self.out(x)
#k = 1
#head_reg4 = SSD_Head(k, -3.)
#models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

drop=0.4

class SSD_MultiHead(nn.Module):
    def __init__(self, k, bias):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.sconv1 = StdConv(256,256, drop=drop)
        self.sconv2 = StdConv(256,256, drop=drop)
        self.sconv3 = StdConv(256,256, drop=drop)
        self.out0 = OutConv(k, 256, bias)
        self.out1 = OutConv(k, 256, bias)
        self.out2 = OutConv(k, 256, bias)
        self.out3 = OutConv(k, 256, bias)

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out1(x)
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        return [torch.cat([o1c,o2c,o3c], dim=1),
                torch.cat([o1l,o2l,o3l], dim=1)]

#head_reg4 = SSD_MultiHead(k, -4.)
#models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_net = FeatureNet()
        self.ssd_head = SSD_Head(1, -3.)
    def forward(self, x):
        p1, p2, p3, p4 = self.feature_net(x)
        out = self.ssd_head(p4)
        return out


def create_model():
    model_detect = DetectionNet().cuda()
    #model_detect.fc = SSD_Head(9, -4.)

    #num_ftrs = model_detect.fc.in_features

    #model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    #model_ft = model_ft.cuda()

    return model_detect

def test_forward():
    model = create_model()
    imgs = torch.randn(2,3,224,224).cuda()
    
    out = model(imgs)
    print(len(out))
    print([x.size() for x in out])

if __name__ == '__main__':
    test_forward()