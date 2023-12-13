import torch

#
# ResNet 20, 36, 44, 56, 110, 1202 for CIFAR10
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016) [1]
#  - https://arxiv.org/abs/1512.03385
#


class ResNetBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBasicBlock, self).__init__()
        # Using the [3x3 , 3x3] x n convention. Section 4.2 [1]
        # Between stacks, the subsampling is performed by convolutions with
        #   a stride of 2. Section 4.2 [1]
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        #
        # strides = 1:
        #   The identity shortcuts (Eqn.(1)) can be directly used when the input
        #   and output are of the same dimensions (solid line shortcuts in Fig. 3).
        # strides = 2:
        #   When the dimensions increase (dotted line shortcuts in Fig. 3), we consider
        #   two options: ... (B) The projection shortcut in Eqn.(2) is used to match
        #   dimensions (done by 1×1 convolutions).
        # Section 3.3 "Residual Network" [1]
        #
        self.downsample = False
        if stride != 1 or in_planes != planes:
            self.downsample = True
            self.shortcut = torch.nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, inputs):
        # Following ResNet building block from Figure 2 [1].
        # We adopt batch normalization right after each convolution and before
        # activation. Section 3.4 [1]
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            # The projection shortcut in Eqn.(2) is used to match dimensions
            # (done by 1×1 convolutions). Section 3.3 "Residual Network" [1]
            y = self.shortcut(inputs)
        else:
            # The identity shortcuts (Eqn.(1)) can be directly used when the input
            # and output are of the same dimensions. Section 3.3 "Residual Network" [1]
            y = inputs

        out = x + y
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        #
        # The following table summarizes the architecture: Section 4.2 [1]
        # | output map size | 32×32 | 16×16 | 8×8 |
        # |-----------------|-------|-------|-----|
        # | # layers        | 1+2n  | 2n    | 2n  |
        # | # filters       | 16    | 32    | 64  |
        #
        # n = num_blocks
        #
        # num_blocks = 3: ResNet20
        # num_blocks = 5: ResNet32
        # num_blocks = 7: ResNet44
        # num_blocks = 9: ResNet56
        # num_blocks = 18: ResNet110
        # num_blocks = 200: ResNet1202
        #
        self.in_planes = 16

        # The first layer is 3×3 convolutions. Section 4.2 [1]
        self.conv1 = torch.nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        # We adopt batch normalization right after each convolution and before
        # activation. Section 3.4 [1]
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()

        # The numbers of filters are {16, 32, 64} respectively. Section 4.2 [1]

        # The first stack uses 16 filters. Section 4.2 [1]
        # First block of the first stack uses identity shortcut (strides=1) since the
        #   input size matches the output size of the first layer is 3×3 convolutions.
        #   Section 3.3 "Residual Network" [1]
        self.conv2 = self._make_layer(ResNetBasicBlock, 16, num_blocks, first_stride=1)
        # The second stack uses 32 filters. Section 4.2 [1]
        # First block of the second stack uses projection shortcut (strides=2)
        #   Section 3.3 "Residual Network" [1]
        self.conv3 = self._make_layer(ResNetBasicBlock, 32, num_blocks, first_stride=2)
        # The third stack uses 64 filters. Section 4.2 [1]
        # First block of the third stack uses projection shortcut (strides=2)
        #   Section 3.3 "Residual Network" [1]
        self.conv4 = self._make_layer(ResNetBasicBlock, 64, num_blocks, first_stride=2)

        # The network ends with a global average pooling, a 10-way fully-connected
        # layer, and softmax. Section 4.2 [1]
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, planes, num_blocks, first_stride):
        layers = []

        # First block of the first stack uses identity shortcut (strides=1) since the
        # First block of the second & third stack uses projection shortcut (strides=2)
        layers.append(block(self.in_planes, planes, first_stride))
        self.in_planes = planes

        for blocks in range(num_blocks - 1):
            # All other blocks use identity shortcut (strides=1)
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
