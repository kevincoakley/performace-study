import torch
import torchvision

#
# DenseNet (k12d40, k12d100, k24d100) & DenseNet-BC (k12d100, k24d250, k40d190) for CIFAR10
#
# Huang, Gao, et al. "Densely connected convolutional networks." (2017) [1]
#  - https://arxiv.org/pdf/1608.06993.pdf
#
# Code from paper:
#  https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua [2]
#


class DenseNetBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, growth_rate, bottleneck):
        super(DenseNetBasicBlock, self).__init__()

        self.bottleneck = bottleneck

        post_bottleneck_in_planes = in_planes

        # A 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution
        # to reduce the number of input feature-maps, and thus to improve computational
        # efficiency. Section 3 "Bottleneck layers" [1]
        if self.bottleneck:
            self.bn_bnk = torch.nn.BatchNorm2d(in_planes, eps=1e-5)
            self.relu_bnk = torch.nn.ReLU(inplace=True)

            # We let each 1×1 convolution produce 4k (4 * growth_rate) feature-maps.
            # Section 3 "Bottleneck layers" [1]
            post_bottleneck_in_planes = 4 * growth_rate
            self.conv2_bnk = torch.nn.Conv2d(
                in_planes,
                post_bottleneck_in_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        # For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel
        # to keep the feature-map size fixed. Section 3 "Implementation Details" [1]
        self.bn = torch.nn.BatchNorm2d(post_bottleneck_in_planes, eps=1e-5)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            post_bottleneck_in_planes,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, inputs):
        if self.bottleneck:
            x = self.bn_bnk(inputs)
            x = self.relu_bnk(x)
            x = self.conv2_bnk(x)
        else:
            x = inputs

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = torch.cat([inputs, x], 1)

        return out


class DenseNetTransitionBlock(torch.nn.Module):
    def __init__(self, in_planes, compression_reduction):
        super(DenseNetTransitionBlock, self).__init__()

        #
        # We use 1×1 convolution followed by 2×2 average pooling as transition layers
        # between two contiguous dense blocks. Section 3 "Implementation Details" [1]
        #
        self.bn1 = torch.nn.BatchNorm2d(in_planes, eps=1e-5)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            in_planes,
            int(in_planes * compression_reduction),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.avgpool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu(x)
        x = self.conv2(x)

        out = self.avgpool(x)

        return out


class DenseNet(torch.nn.Module):
    def __init__(
        self, depth, growth_rate, compression_reduction, bottleneck, num_classes=10
    ):
        super(DenseNet, self).__init__()

        # Dense connected layers in each denseblock (DenseNet) [2]
        n = int((depth - 4) / 3)

        # See Section 3 "Implementation Details" [1]
        input_filter = 16

        if bottleneck:
            # See Section 3 "Implementation Details" [1]
            input_filter = growth_rate * 2
            # Dense connected layers in each denseblock (DenseNet-BC) [2]
            n = int(n / 2)

        # Track the growth after each dense block
        self.n_channels = input_filter

        # Before entering the first dense block, a convolution with 16 (or twice the growth
        # rate for DenseNet-BC). Section 3 "Implementation Details" [1] & [2]
        self.conv1 = torch.nn.Conv2d(
            3, self.n_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        #
        # DenseNet used in our experiments has three dense blocks that each has
        # an equal number of layers. Section 3 "Implementation Details" [1]
        #

        # First dense block
        self.conv2 = self._make_layer(DenseNetBasicBlock, n, growth_rate, bottleneck)

        # First transition block
        self.trans1 = DenseNetTransitionBlock(self.n_channels, compression_reduction)
        # To further improve model compactness, we can reduce the number of feature-maps at
        # transition layers. When θ = 1, the number of feature-maps across transition layers
        # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
        # in our experiment. Section 3 "Compression" [1]
        self.n_channels = int(self.n_channels * compression_reduction)

        # Second dense block
        self.conv3 = self._make_layer(DenseNetBasicBlock, n, growth_rate, bottleneck)

        # Second transition block
        self.trans2 = DenseNetTransitionBlock(self.n_channels, compression_reduction)
        # To further improve model compactness, we can reduce the number of feature-maps at
        # transition layers. When θ = 1, the number of feature-maps across transition layers
        # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
        # in our experiment. Section 3 "Compression" [1]
        self.n_channels = int(self.n_channels * compression_reduction)

        # Third dense block
        self.conv4 = self._make_layer(DenseNetBasicBlock, n, growth_rate, bottleneck)

        # Last transition block before the classifier. Only use BN-ReLU after the last
        # dense block. See addTransition() [2]
        self.bn1 = torch.nn.BatchNorm2d(self.n_channels, eps=1e-5)
        self.relu = torch.nn.ReLU(inplace=True)

        # At the end of the last dense block, a global average pooling is performed and
        # then a softmax classifier is attached. Section 3 "Implementation Details" [1]
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.n_channels, num_classes)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, num_blocks, growth_rate, bottleneck):
        layers = []

        for blocks in range(num_blocks):
            layers.append(block(self.n_channels, growth_rate, bottleneck))
            # Each layer adds k feature-maps of its own to this state. The growth rate
            # regulates how much new information each layer contributes to the global state.
            # Section 3 "Growth rate" [1]
            self.n_channels += growth_rate

        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)

        x = self.conv2(x)
        x = self.trans1(x)

        x = self.conv3(x)
        x = self.trans2(x)

        x = self.conv4(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def densenet_k12d40(num_classes=10):
    return DenseNet(
        depth=40,
        growth_rate=12,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_k12d100(num_classes=10):
    return DenseNet(
        depth=100,
        growth_rate=12,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_k24d100(num_classes=10):
    return DenseNet(
        depth=100,
        growth_rate=24,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_bc_k12d100(num_classes=10):
    return DenseNet(
        depth=100,
        growth_rate=12,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


def densenet_bc_k24d250(num_classes=10):
    return DenseNet(
        depth=250,
        growth_rate=24,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


def densenet_bc_k40d190(num_classes=10):
    return DenseNet(
        depth=190,
        growth_rate=40,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    from torchsummary import summary

    model = densenet_k12d40()
    summary(model, (3, 32, 32))

    from torchview import draw_graph

    batch_size = 64
    model_graph = draw_graph(
        model, input_size=(batch_size, 3, 32, 32), save_graph=True, device="meta"
    )
