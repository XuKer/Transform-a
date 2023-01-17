import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath,trunc_normal_
class Mlp(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super(Mlp, self).__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x
class Attention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qkv_scale=None,atten_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qkv_scale or head_dim** -0.5
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.atten_drop=nn.Dropout(atten_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        atten=(q @ k.transpose(-2,-1))*self.scale
        atten=atten.softmax(dim=-1)
        atten=self.atten_drop(atten)
        x=(atten @ v).transpose(1,2).reshape(B,N,C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x
class Block(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,qk_scale=None,drop=0.,attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU,norm_layer=partial(nn.LayerNorm,eps=1e-6)):
        super().__init__()
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,num_heads,qkv_bias=qkv_bias,qkv_scale=qk_scale,atten_drop=attn_drop,
                            proj_drop=drop)
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2=norm_layer(dim)
        mlp_hidden_dim=int(dim*mlp_ratio)
        self.mlp=Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x
class ConvBlock(nn.Module):
    def __init__(self,inplanes,outplanes,stride=1,res_conv=False,act_layer=nn.ReLU,groups=1,
                 norm_layer=partial(nn.BatchNorm2d,eps=1e-6),drop_block=None,drop_path=None):
        super(ConvBlock, self).__init__()

        expansion=4
        med_planes=outplanes//expansion
        self.conv1=nn.Conv2d(inplanes,med_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1=norm_layer(med_planes)
        self.act1=act_layer(inplanes=True)
        self.conv2=nn.Conv2d(med_planes,med_planes,kernel_size=3,stride=stride,groups=groups,padding=1,
                             bias=False)
        self.bn2=norm_layer(med_planes)
        self.act2=act_layer(inplace=True)

        self.conv3=nn.Conv2d(med_planes,outplanes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=norm_layer(outplanes)
        self.act3=act_layer(inplace=True)

        if res_conv:
            self.residual_conv=nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=stride,padding=0,bias=False)
            self.residual_bn=norm_layer(outplanes)

        self.res_conv=res_conv
        self.drop_block=drop_block
        self.drop_path=drop_path
    def zero_init_last_bn(self):
        nn.init.zeros_((self.bn3.weight))
    def forward(self,x,x_t=None,return_x_2=True):
        residual=x

        x=self.conv1(x)
        x=self.bn1(x)
        if self.drop_block is not None:
            x=self.drop_block(x)
        x=self.act1(x)

        x=self.conv2(x) if x_t is None else self.conv2(x+x_t)
        x=self.bn2(x)
        if self.drop_block is not None:
            x=self.drop_block(x)
        x2=self.act2(x)

        x=self.conv3(x2)
        x=self.bn3(x)
        if self.drop_block is not None:
            x=self.drop_block(x)

        if self.drop_path is not None:
            x=self.drop_path(x)

        if self.res_conv:
            residual=self.residual_conv(residual)
            residual=self.residual_bn(residual)
        x+=residual
        x=self.act3(x)

        if return_x_2:
            return x,x2
        else:
            return x
        # cnn to transformer
class FCUDown(nn.Module):
    def __init__(self,inplanes,outplanes,dw_stride,act_layer=nn.GELU
                 ,norm_layer=partial(nn.LayerNorm,eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride=dw_stride
        self.conv_project=nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,padding=0)
        self.sample_pooling=nn.AvgPool2d(kernel_size=dw_stride,stride=dw_stride)

        self.ln=norm_layer(outplanes)
        self.act=act_layer()

    def forward(self,x,x_t):
        x=self.conv_project(x)
        x=self.sample_pooling(x).flatten(2).transpose(1,2)
        x=self.ln(x)
        x=self.act(x)

        x=torch.cat([x_t[:,0][:,None,:],x],dim=1)

        return x


class FCUUP(nn.Module):
    def __init__(self,inplanes,outplanes,up_straide,act_layer=nn.ReLU,norm_layer=partial(nn.BatchNorm2d,
                                                                                         eps=1e-6)):
        super(FCUUP, self).__init__()
        self.up_stride=up_straide
        self.conv_project=nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn=norm_layer(outplanes)
        self.act=act_layer()

    def forward(self,x,H,W):
        B,_,C=x.shape

        x_r=x[:1,1:].transpose(1,2).reshape(B,C,H,W)
        x_r=self.act(self.bn(self.conv_project(x_r)))


        return F.interpolate(x_r,size=(H * self.up_stride,W * self.up_stride))
class Med_ConvBlock(nn.Module):
    def __init__(self,inplanes,act_layer=nn.ReLU,groups=1,norm_layer=partial(nn.BatchNorm2d,eps=1e-6),
                 drop_block=None,drop_path=None):
        super(Med_ConvBlock, self).__init__()
        expansion=4
        med_planes=inplanes//expansion

        self.conv1=nn.Conv2d(inplanes,med_planes,kernel_size=1,stride=1,padding=,bias=False)
        self.bn1=norm_layer(med_planes)
        self.act1=act_layer(inplace=True)

        self.conv2=nn.Conv2d(med_planes,med_planes,kernel_size=3,stride=1,groups=groups,
                             padding=1,bias=False)
        self.bn2=norm_layer(med_planes)
        self.act2=act_layer(inplace=True)

        self.conv3=nn.Conv2d(med_planes,inplanes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=norm_layer(inplanes)
        self.act3=act_layer(inplace=True)
        self.drop_block=drop_block
        self.drop_path=drop_path
    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)
    def forward(self,x):
        residual=x

        x=self.conv1(x)
        x=self.bn1(x)
        if self.drop_block is not None:
            x=self.drop_block(x)
        x=self.act1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        if self.drop_block is not None:
            x=self.drop_block(x)
        x=self.act2(x)

        x=self.conv3(x)
        x=self.bn3(x)
        if self.drop_block is not None:
            x=self.drop_block(x)

        if self.drop_path is not None:
            x=self.drop_path(x)
        x+=residual
        x=self.act3(x)

        return  x
class ConvTransBlock(nn.Module):































