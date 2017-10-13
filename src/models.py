"""
@author Konstantin Lopuhin
"""

from functools import partial

from torch import nn
import torchvision.models as M
import torch.functional as F
import torch.nn.functional

resnet18 = M.resnet18
resnet34 = M.resnet34
resnet50 = M.resnet50
resnet101 = M.resnet101
resnet152 = M.resnet152
vgg16 = M.vgg16
vgg16_bn = M.vgg16_bn
densenet121 = M.densenet121
densenet161 = M.densenet161
densenet201 = M.densenet201


class AvgPool(nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, (x.size(2), x.size(3)))


class ResNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.avgpool = AvgPool()

        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNetFinetune(nn.Module):
    finetune = True

    def __init__(self, num_classes, net_cls=M.densenet121, two_layer=False):
        super().__init__()
        self.net = net_cls(pretrained=True)
        self.net.avgpool = AvgPool()
        self.relu = torch.nn.ReLU(inplace=True)

        if two_layer:
            mid_channels = 512
            self.net.classifier = nn.Sequential(
                nn.Linear(self.net.classifier.in_features, mid_channels),
                nn.Dropout(),
                nn.Linear(mid_channels, num_classes))
        else:
            self.net.classifier = nn.Linear(
                self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        net = self.net
        features = net.features(x)
        out = self.relu(features)
        out = torch.nn.functional.avg_pool2d(out, (out.size(2), out.size(3)))
        out = out.view(features.size(0), -1)
        out = net.classifier(out)
        return out
    #
    #
    # def forward(self, x):
    #     return self.net(x)


class InceptionV3Finetune(nn.Module):
    finetune = True

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = M.inception_v3(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


resnet34_finetune = partial(ResNetFinetune, net_cls=M.resnet34)
resnet50_finetune = partial(ResNetFinetune, net_cls=M.resnet50)
resnet101_finetune = partial(ResNetFinetune, net_cls=M.resnet101)
resnet152_finetune = partial(ResNetFinetune, net_cls=M.resnet152)

densenet121_finetune = partial(DenseNetFinetune, net_cls=M.densenet121)
densenet121_finetune_2l = partial(DenseNetFinetune, net_cls=M.densenet121,
                                  two_layer=True)

densenet161_finetune = partial(DenseNetFinetune, net_cls=M.densenet161)
densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)

inceptionv3_finetune = InceptionV3Finetune
