import torch

#
# ResNet 20, 36, 44, 56, 110, 1202 for CIFAR10
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016)
#  - https://arxiv.org/abs/1512.03385
#


class ResNetBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBasicBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes, eps=1e-5, momentum=0.1)

        self.downsample = False
        if stride != 1 or in_planes != planes:
            self.downsample = True
            self.shortcut = torch.nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, inputs):
        y = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            y = self.shortcut(inputs)

        out = x + y
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = torch.nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = self._make_layer(ResNetBasicBlock, 16, num_blocks, first_stride=1)
        self.conv3 = self._make_layer(ResNetBasicBlock, 32, num_blocks, first_stride=2)
        self.conv4 = self._make_layer(ResNetBasicBlock, 64, num_blocks, first_stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, first_stride):
        layers = []

        layers.append(block(self.in_planes, planes, first_stride))
        self.in_planes = planes

        for blocks in range(num_blocks - 1):
            layers.append(block(self.in_planes, planes, 1))

        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def resnet20(num_classes=10):
    return ResNet(3, num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(5, num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(7, num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet(9, num_classes=num_classes)


def resnet110(num_classes=10):
    return ResNet(18, num_classes=num_classes)


def resnet1202(num_classes=10):
    return ResNet(200, num_classes=num_classes)


if __name__ == "__main__":
    from torchsummary import summary

    model = resnet20()
    summary(model, (3, 32, 32))

    from torchview import draw_graph

    batch_size = 128
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 32, 32), save_graph=True, device="meta"
    )
