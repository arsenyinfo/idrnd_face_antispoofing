from pretrainedmodels.models import densenet121, densenet201, nasnetalarge, se_resnet50, se_resnext101_32x4d
from torch import nn
from torch.nn import functional as F


class Baseline(nn.Module):
    def __init__(self, backbone_fn):
        super().__init__()
        model = backbone_fn(pretrained='imagenet')
        self.backbone = model.features
        self.linear = nn.Linear(model.last_linear.in_features, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.linear(x)
        return x


def get_baseline(name='densenet121'):
    backbones = {'densenet121': densenet121,
                 'densenet201': densenet201,
                 'se_resnet50': se_resnet50,
                 'se_resnext101': se_resnext101_32x4d,
                 'nasnet': nasnetalarge,
                 }
    return Baseline(backbones[name])
