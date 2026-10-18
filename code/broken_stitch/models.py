"""Approximate 2022 CNN/VGG-16 model families (not recovered historical weights)."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights


def activation(name):
    # Softmax used as a hidden activation is not standard; retained ONLY as a
    # documented historical activation-ablation approximation.
    return {'relu':nn.ReLU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,
            'selu':nn.SELU,'elu':nn.ELU,'softmax':lambda:nn.Softmax(dim=1)}[name]()


class SmallCNN(nn.Module):
    def __init__(self,activation_name='relu'):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),activation(activation_name),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),activation(activation_name),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),activation(activation_name),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128,64),activation(activation_name),nn.Dropout(.25),nn.Linear(64,2))
    def forward(self,x):return self.head(self.features(x))


class VGGStitch(nn.Module):
    def __init__(self,features,finetune=False):
        super().__init__()
        self.features=features
        self.pool=nn.AdaptiveAvgPool2d(1)
        if finetune:
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,256),nn.ReLU(inplace=True),nn.Dropout(.25),nn.Linear(256,2))
        else:self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,2))
        for p in self.features.parameters():p.requires_grad=False
        if finetune:
            # Only last VGG block is trainable. This is a declared reconstruction
            # choice; exact old fine-tuned layers cannot be recovered from thesis.
            for p in self.features[24:].parameters():p.requires_grad=True
    def forward(self,x):return self.head(self.pool(self.features(x)))


def _vgg_features(weights_path=None,allow_download=False,allow_random=False):
    # Do not silently call "transfer learning" when pretrained weights absent.
    if weights_path is None and not allow_download and not allow_random:
        raise ValueError('VGG needs --vgg-weights /path/to/torchvision_vgg16_weights.pth or --allow-download. '
                         '--allow-random-vgg is for smoke tests only (NOT publishable pretrained results).')
    if allow_download and weights_path is None:
        backbone=vgg16(weights=VGG16_Weights.DEFAULT)
        return backbone.features,'torchvision_ImageNet1K_V1_download'
    # Construct features alone so we avoid allocating the huge VGG classification head.
    features=vgg16(weights=None).features
    if weights_path:
        f=Path(weights_path)
        if not f.is_file(): raise FileNotFoundError(f'VGG weights not found: {f}')
        raw=torch.load(f,map_location='cpu',weights_only=True)
        if isinstance(raw,dict) and 'state_dict' in raw:raw=raw['state_dict']
        if not isinstance(raw,dict):raise ValueError('Expected a state_dict (.pth)')
        if any(key.startswith('features.') for key in raw):
            weights={k[len('features.'):]:v for k,v in raw.items() if k.startswith('features.')}
        else: weights=raw
        features.load_state_dict(weights,strict=True)
        return features,f'local:{f.name}'
    return features,'RANDOM_UNTRAINED_VGG_NOT_FOR_PUBLICATION'


def create_model(name,activation_name='relu',vgg_weights=None,allow_download=False,allow_random_vgg=False):
    if name=='cnn':return SmallCNN(activation_name),'scratch_random_init'
    if name in ('vgg_linear','vgg_finetune'):
        features,provenance=_vgg_features(vgg_weights,allow_download,allow_random_vgg)
        return VGGStitch(features,finetune=name=='vgg_finetune'),provenance
    raise ValueError('Choose cnn, vgg_linear or vgg_finetune')