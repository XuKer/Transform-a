import torch
from torch import nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t,tuple) else (t,t)


class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)