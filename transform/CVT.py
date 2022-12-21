import torch
from torch import nn,einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

def group_dict_by_key(cond,d):
    return_val=[dict(),dict()]
    for key in d.keys():
        match=bool(cond(key))
        ind=int(not match)
        return_val[ind][key]=d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix,d):
    kwargs_with_prefix,kwargs=group_dict_by_key(lambda x: x.startswith(prefix),d)
    kwargs_without_prefix=dict(map(lambda x:(x[0][len(prefix):],x[1]),tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix,kwargs


#   类
class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.g=nn.Parameter(torch.ones(1,dim,1,1))
        self.b=nn.Parameter(torch.zeros(1,dim,1,1))
    def forward(self,x):
        var=torch.var(x,dim=1,unbiased=False,keepdim=True)
        mean=torch.mean(x,dim=1,keepdim=True)
        return (x-mean)/(var+self.eps).sqrt()*self.g+self.b

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm=LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        x=self.norm(x)
        return self.fn(x,**kwargs)
class FeedForward(nn.Module):
    def __init__(self,dim,mult=4,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(dim,dim*mult,1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim*mult,dim,1),
            nn.Dropout(dropout)

        )
    def forward(self,x):
        return self.net(x)
class DepthWiseConv2d(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size,padding,stride,bias=True):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(dim_in,dim_in,kernel_size=kernel_size,padding=padding,groups=dim_in,stride=stride,bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in,dim_out,kernel_size=1,bias=bias)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,proj_kernel,kv_proj_stride,heads=8,dim_head=64,dropout=0.):
        super().__init__()
        inner_dim=dim_head*heads
        padding=proj_kernel//2
        self.heads=heads
        self.scale=dim_head**-0.5
        self.attend=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(dropout)

        self.to_q=DepthWiseConv2d(dim,inner_dim,proj_kernel,padding=padding,stride=1,bias=False)
        self.to_kv=DepthWiseConv2d(dim,inner_dim*2,proj_kernel,padding=padding,stride=kv_proj_stride,bias=False)
        self.to_out=nn.Sequential(
            nn.Conv2d(inner_dim,dim,1),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        shape=x.shape







