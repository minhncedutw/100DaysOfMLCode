from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

"""
Default distribute a lot of layers/blocks into 3 groups
ConvLearner <- ConvnetBuilder.get_layer_groups <- split_by_idxs
ConvLearner.pretrained(...)
"""
PATH = "E:/WORKSPACES/python/DATA/dogscats/"
sz = 224
arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(f=arch, data=data, precompute=True)
learn.summary()

lr = 1e-2
learn.fit(lr, 1)

learn.unfreeze() # Must unfreeze network to be able to train multiple-learning-rate
print("Number of groups: ", len(learn.models.get_layer_groups()))
lrs = np.array([lr/100, lr/10, lr]) # The number of lr values must be equal to the number of groups
learn.fit(lrs, 1, cycle_len=1, cycle_mult=2)

"""
Default distribute a lot of layers into a lot of single groups
ConvLearner <- BasicModel.get_layer_groups <- architecture
ConvLearner(...)
"""
class SimpleConv(nn.Module):
    def __init__(self, ic, oc, ks=3, drop=0.2, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, ks, padding=(ks - 1) // 2)
        self.bn = nn.BatchNorm2d(oc, momentum=0.05) if bn else None
        self.drop = nn.Dropout(drop, inplace=True)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        return self.drop(self.act(x))

net = nn.Sequential(
    SimpleConv(3, 64),
    SimpleConv(64, 128),
    SimpleConv(128, 128),
    SimpleConv(128, 128),
    nn.MaxPool2d(2),
    SimpleConv(128, 128),
    SimpleConv(128, 128),
    SimpleConv(128, 256),
    nn.MaxPool2d(2),
    SimpleConv(256, 256),
    SimpleConv(256, 256),
    nn.MaxPool2d(2),
    SimpleConv(256, 512),
    SimpleConv(512, 2048, ks=1, bn=False),
    SimpleConv(2048, 256, ks=1, bn=False),
    nn.MaxPool2d(2),
    SimpleConv(256, 256, bn=False, drop=0),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(256, 10)
)

PATH = Path("./data/cifar10/")
bs = 64
sz = 32
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

bm = BasicModel(net.cuda(), name='simplenet')
learn = ConvLearner(data, bm)
learn.crit = nn.CrossEntropyLoss()
learn.opt_fn = optim.Adam
learn.unfreeze()
learn.metrics=[accuracy]
lr = 1e-3
wd = 5e-3
learn.fit(lr, 1)

print("Number of groups: ", len(learn.models.get_layer_groups()))
lrs = np.array([lr/9, lr/9, lr/9, lr/9, lr/9, lr/9, lr/9, lr/9, lr/9,
                lr/3, lr/3, lr/3, lr/3, lr/3, lr/3, lr/3, lr/3, lr/3, lr/3,
                lr])
learn.fit(lrs, 1)

"""
Divide the network of 2 layers/blocks into 2 groups
ConvLearner <- BasicModel.get_layer_groups <- architecture
ConvLearner.from_model_data(...)
"""
class ConvNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2) for i in range(len(layers) - 1)]) # block 1: multi-layers
        self.out = nn.Linear(layers[-1], c) # block 2: 1 layer

    def forward(self, x):
        for l in self.layers: x = F.relu(l(x))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

PATH = Path("./data/cifar10/")
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)
data = get_data(32,4)

arch = ConvNet([3, 20, 40, 80], 10)
learn = ConvLearner.from_model_data(arch, data)
lr = 1e-3
learn.fit(lr, 1)
print("Number of groups: ", len(learn.models.get_layer_groups()))
lrs = np.array([lr/2, lr])
learn.fit(lrs, 1)

"""
"""
class BnLayer(nn.Module):
    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, bias=False, padding=1)
        self.a = nn.Parameter(torch.zeros(nf, 1, 1))
        self.m = nn.Parameter(torch.ones(nf, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x_chan = x.transpose(0, 1).contiguous().view(x.size(1), -1)
        if self.training:
            self.means = x_chan.mean(1)[:, None, None]
            self.stds = x_chan.std(1)[:, None, None]
        return (x - self.means) / self.stds * self.m + self.a


class ConvBnNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv1(x)
        for l in self.layers: x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

learn = ConvLearner.from_model_data(ConvBnNet([10, 20, 40, 80, 160], 10), data)
print("Number of groups: ", len(learn.models.get_layer_groups())) # "Number of groups:  3"

"""
"""
class ResnetLayer(BnLayer):
    def forward(self, x): return x + super().forward(x)

class Resnet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i + 1])
                                     for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i + 1], layers[i + 1], 1)
                                      for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i + 1], layers[i + 1], 1)
                                      for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv1(x)
        for l, l2, l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

learn = ConvLearner.from_model_data(Resnet([10, 20, 40, 80, 160], 10), data)
print("Number of groups: ", len(learn.models.get_layer_groups())) # "Number of groups:  5"

"""
"""
class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, kernel_size=3):
        super().__init__()
        self.BN = BnLayer(ni, nf)
        self.Res1 = ResnetLayer(nf, nf, stride=stride)
        self.Res2 = ResnetLayer(nf, nf, stride=stride)

    def forward(self, x):
        return self.Res2(self.Res1(self.BN(x)))

class Resnet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([ResBlock(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv1(x)
        for l in self.layers:
            x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

learn = ConvLearner.from_model_data(Resnet2([10, 20, 40, 80, 160], 10), data)
print("Number of groups: ", len(learn.models.get_layer_groups())) # "Number of groups:  5"
lr = 1e-3
lrs = np.array([lr/4, lr/2, lr])
learn.fit(lrs, 1)