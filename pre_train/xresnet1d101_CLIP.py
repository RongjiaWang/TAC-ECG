
import datetime
import torch
from torch import nn
import torch.utils.tensorboard
from transformers import BertModel
import numpy as np
import inspect

"""搭建模型"""
"""搭建图像特征提取器"""
def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f

def AvgPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.AvgPool1d(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


def MaxPool(ks=2, stride=None, padding=0, ceil_mode=False):
    return nn.MaxPool1d(ks, stride=stride, padding=padding)


def AdaptiveAvgPool(sz=1):
    return nn.AdaptiveAvgPool1d(sz)


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class ConvLayer(nn.Sequential):
    """
    Creates a sequence of Conv, Act, Norm
    """

    @delegates(nn.Conv1d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, norm='bn', bn_1st=True,
                 act_cls=nn.ReLU, xtra=None, **kwargs):
        if padding is None: padding = ((ks - 1) // 2)
        norm = nn.BatchNorm1d(nf)
        bias = None if not (not norm) else bias
        conv = nn.Conv1d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if norm: act_bn.append(norm)
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

class ResBlock(nn.Module):
    """
    Resnet block from ni to nh with stride
    """

    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, nh1=None, nh2=None,
                 norm='bn', act_cls=nn.ReLU, ks=3, pool_first=True, **kwargs):
        super(ResBlock, self).__init__()
        norm1 = norm2 = norm
        pool = AvgPool
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm=norm1, act_cls=act_cls, **kwargs)
        k1 = dict(norm=norm2, act_cls=None, **kwargs)
        conv_path = [
            ConvLayer(ni, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, ks, **k1)
        ] if expansion == 1 else [
            ConvLayer(ni, nh1, 1, **k0),
            ConvLayer(nh1, nh2, ks, stride=stride, **k0),
            ConvLayer(nh2, nf, 1, **k1)]
        self.conv_path = nn.Sequential(*conv_path)
        id_path = []
        if ni != nf: id_path.append(ConvLayer(ni, nf, 1, norm=norm, act_cls=None, **kwargs))
        if stride != 1: id_path.insert((1, 0)[pool_first], pool(stride, ceil_mode=True))
        self.id_path = nn.Sequential(*id_path)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.conv_path(x) + self.id_path(x))

class XResNet(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, input_channels=12, num_classes=5, stem_szs=(32, 32, 64),
                 widen=1.0, norm='bn', act_cls=nn.ReLU, ks=3, stride=2, **kwargs):
        self.block, self.expansion, self.act_cls, self.ks = block, expansion, act_cls, ks
        if ks % 2 == 0: raise Exception('Kernel size has to be odd')
        self.norm = norm
        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(stem_szs[i], stem_szs[i + 1], ks=ks, stride=stride if i == 0 else 1, norm=norm, act_cls=act_cls)
            for i in range(3)]
        # block_szs = [int(o * widen) for o in [64, 128, 256, 512] + [256] * (len(layers) - 4)]
        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs
        blocks = self._make_blocks(layers, block_szs, stride, **kwargs)

        # head = head_layer(inplanes=block_szs[-1] * expansion, ps_head=0.5, num_classes=num_classes)

        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks // 2),
            *blocks,
            # head,
            AdaptiveAvgPool(sz=1), Flatten(), nn.Dropout(p),
            # nn.Linear(block_szs[-1] * expansion, num_classes),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i + 1], blocks=l,
                                 stride=1 if i == 0 else stride, **kwargs)
                for i, l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i == 0 else nf, nf, stride=stride if i == 0 else 1,
                         norm=self.norm, act_cls=self.act_cls, ks=self.ks, **kwargs)
              for i in range(blocks)])

def xresnet1d101(**kwargs):
    return XResNet(ResBlock, 4, [3, 4, 23, 3], input_channels=12, **kwargs)

def xresnet1d50(**kwargs):
    return XResNet(ResBlock, 4, [3, 4, 6, 3], input_channels=12, **kwargs)

"""搭建图像特征与文本特征的尺寸适配器"""
class MLP(nn.Module):
    def __init__(self):
        super(adapter, self).__init__()
        self.linear = nn.Linear(in_features=768, out_features=256)

    def forward(self, x):
        return self.linear(x)

"""搭建CLIP模型"""
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()

        self.visual = xresnet1d101()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.MLP = MLP()

    @property
    def dtype(self):
        return self.MLP.linear.weight.dtype

    """搭建图像特征提取器"""
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    """搭建文本特征提取器"""
    def encode_text(self, text):
        # 输入[batch_size, 标签个数*max_length]
        bert_model = BertModel.from_pretrained('/Bio_ClinicalBERT')
        bert_model.to(device)
        outputs = bert_model(text)
        # 获取最后一个隐藏层表示,其形状为 [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state  # 形状为[batch_size, 标签个数*max_length, 768]
        # 若last_hidden_state的第0维是1，则删除这个维度
        # token_embeddings = torch.squeeze(last_hidden_state, dim=0)  # 形状为[batch_size, 标签个数*max_length, 768]
        token_embeddings = last_hidden_state
        # 选取张量 token_embeddings 中每行中最大值对应的元素
        token_embeddings_index = token_embeddings[torch.arange(token_embeddings.shape[0]), text.argmax(dim=-1)]
        # 形状为[batch_size, 768]
        # 与文本维度相匹配
        text_features = self.MLP(token_embeddings_index)  # 形状为[batch_size, 128]

        return text_features

    def forward(self, image, text):

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 规范化特征,# 每一行sqr(a1^2+a2^2+...)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [batch_img,128]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [batch_text,128]

        # 计算相似度
        logit_scale = self.logit_scale.exp()  # 可学习参数
        logits_per_image = logit_scale * image_features @ text_features.t()  # 特征相乘获得相似度
        logits_per_text = logits_per_image.t()  # 变成文本

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

