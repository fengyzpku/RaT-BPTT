import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)
        
class MLP(nn.Module):
    def __init__(self, n, width, num_classes=10):
        super(MLP, self).__init__()
        layer_list = [Flatten(), nn.Linear(28*28, width), nn.BatchNorm1d(width), nn.ReLU()]
        for i in range(n):
            layer_list.append(nn.Linear(width, width))
            #layer_list.append(nn.BatchNorm1d(width))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(width, num_classes))
        self.MLP = nn.Sequential(
            *layer_list
        )
        self.apply(_weights_init)

    def forward(self, x):
        out = self.MLP(x)
        return out, out


def mlp(n, width):
    return MLP(n, width)

class Linear(nn.Module):
    def __init__(self, num_classes=10):
        super(Linear, self).__init__()
        layer_list = [Flatten(), nn.Linear(28*28, num_classes)]
        self.linear = nn.Sequential(*layer_list)
        self.apply(_weights_init)
        
    def forward(self, x):
        out = self.linear(x)
        return out, out
    
def linear():
    return Linear()