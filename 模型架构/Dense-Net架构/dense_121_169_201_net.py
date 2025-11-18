import re
from matplotlib.pyplot import isinteractive
import torch
import torchvision
from torch import nn
import torch.utils.checkpoint as cp
from typing import List, Tuple
from collections import OrderedDict
from torch.hub import load_state_dict_from_url


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_rate, memory_efficient: bool):
        super().__init__()
        self.memory_efficient = memory_efficient
        self.dropout_rate = dropout_rate
        # 用于bottleneck的forward计算，标号2表示加载checkpoints
        # 这里用先标识了每个层的类型
        self.norm1: nn.BatchNorm2d
        self.relu1: nn.ReLU
        self.conv1: nn.Conv2d
        self.norm2: nn.BatchNorm2d
        self.relu2: nn.ReLU
        self.conv2: nn.Conv2d
        self.drop: nn.Dropout2d
        # 添加bottleneck到网络中
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(
            in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
                                           growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        if self.dropout_rate > 0:
            self.add_module('drop', nn.Dropout2d(self.dropout_rate))

    def bottleneck(self, input: List[torch.Tensor]):
        concated_features = torch.cat(input, dim=1)
        bottle_neck_outputs = self.conv1(
            self.relu1(self.norm1(concated_features)))
        return bottle_neck_outputs

    @torch.jit.unused
    def call_checkpoints_bottleneck(self, input: List[torch.Tensor]):
        def closure(*inputs):
            return self.bottleneck(inputs)

        return cp.checkpoint(closure, *input)

    def forward(self, input: torch.Tensor):
        # 若输入不是list，则转换为list
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        if self.memory_efficient:
            bottleneck_output = self.call_checkpoints_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.dropout_rate > 0:
            new_features = self.drop(new_features)
        return new_features


class _DenseBlock(nn.ModuleDict):
    '''
    stacked dense layers to form a dense block
    '''

    def __init__(self, num_layers, in_channels, growth_rate, bn_size, dropout_rate, memory_efficient) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + growth_rate * i,
                                growth_rate, bn_size, dropout_rate, memory_efficient)
            # 层的标识下标是从1开始，计算size时使用0开始会更方便
            self.add_module(f'denselayer {i + 1}', layer)

    def forward(self, x: torch.Tensor):
        # 先将上个denseblock放入一个列表，然后逐渐添加各denselayer的输出
        # features 会在每个denselayer中的bottlelayer进行concat，然后再进行计算
        # 这样通过denselayer中的checkpoints模块函数进行管理，可以实现memory efficient
        features = [x]
        # self.items()以OrderDict的方式访问self._modules中的layers
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Module):
    '''
    transition layer
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        # 调整channels
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        # 调整feature map的大小
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out


class Densenet(nn.Module):
    def __init__(self, block_config: Tuple[int, int, int, int],
                 num_classes: int = 1000,
                 in_channels: int = 64,
                 growth_rate: int = 32,
                 bn_size: int = 4,
                 dropout_rate: float = 0,
                 memory_efficient: bool = False):
        super().__init__()
        # stage 1: initial convolution
        # 适应fashion mnist，改为单通道
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, in_channels,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(in_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        # stage 2: dense blocks
        num_features = in_channels
        for i, num_layers in enumerate(block_config):
            denseblock = _DenseBlock(
                num_layers, num_features, growth_rate, bn_size, dropout_rate, memory_efficient)
            self.features.add_module(f'denseblock {i + 1}', denseblock)
            num_features += num_layers * growth_rate
            # 判断是否到了最后一层，如果是最后一层，这里应该接分类层，而不是transition layer
            if i != len(block_config) - 1:
                # 这里设置通道数目直接减半，feature map H W同时减半
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition {i + 1}', trans)
                num_features = num_features // 2
        # 结尾前的batchnorm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.features.add_module('flat', nn.Flatten())

        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


class Constructor:
    def __init__(self, num_classes: int = 1000,
                 memory_efficient: bool = False,
                 load: bool = False,
                 progress: bool = True):
        self.num_classes = num_classes
        self.memory_efficient = memory_efficient
        self.load = load
        self.progress = progress
        # 并不能直接用这些官方模型参数，因为模型上有些细节和官方不一样
        # 这里只为了了解官方加载的代码方式而写
        self.model_urls = {
            'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
            'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
            'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        }

    def _load_state_dict(self, model: nn.Module, model_url: str):
        state_dict = load_state_dict_from_url(
            model_url, progress=self.progress)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return model.load_state_dict(state_dict)

    def _build_model(self, block_config, model_url=None):
        model = Densenet(block_config, self.num_classes,
                         memory_efficient=self.memory_efficient)
        if self.load:
            if model_url is None:
                model_url = self.model_urls['densenet121']
            self._load_state_dict(model, model_url)
        return model

    def densenet121(self):
        return self._build_model((6, 12, 24, 16), self.model_urls['densenet121'])

    def densenet169(self):
        return self._build_model((6, 12, 32, 32), self.model_urls['densenet169'])

    def densenet201(self):
        return self._build_model((6, 12, 48, 32), self.model_urls['densenet201'])


from torchinfo import summary
from d2l import torch as d2l

num_classes = 10
memory_efficient = True
load = False
progress = True

densenet121 = Constructor(num_classes, memory_efficient, load,
                          progress).densenet169().to('cpu')
summary(densenet121, input_size=(256, 3, 224, 224), device='cpu')
# X = torch.randn(1, 1, 224, 224).to('cpu')
# print(densenet121(X).shape)