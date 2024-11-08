import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=5):
        super(MyModel, self).__init__()

        if backbone == 'resnet18':
            self.model = models.resnet18(num_classes=num_classes)
        elif backbone == 'resnet34':
            self.model = models.resnet34(num_classes=num_classes)
        else:
            self.model = models.resnet50()

    def forward(self, x):
        return self.model(x)


