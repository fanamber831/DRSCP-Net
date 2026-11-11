from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
from einops import rearrange

from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


import numbers



import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL

class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def forward(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(nn.Module):
    def __init__(self, feature=True, vae=None):
        super().__init__()
        self.model = ControlNetHED_Apache2()
        self.model.eval()
        self.model.requires_grad_(False)
        if feature:
            if vae is None:
                self.vae = AutoencoderKL.from_pretrained("/root/data1/daclip_MRI/pretrained_models/PixArt_VAE")
            else:
                self.vae = vae
            self.vae.eval()
            self.vae.requires_grad_(False)
        else:
            self.vae = None

    def forward(self, input_image):
        B, C, H, W = input_image.shape
        with torch.inference_mode():
            edges = self.model(input_image * 255.)
            edges = torch.cat([TF.resize(e, [H, W]) for e in edges], dim=1)
            edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1, keepdim=True)))
            edge.clip_(0, 1)                                # ([2, 1, 256, 256])
            if self.vae:
                edge = TF.normalize(edge, [.5], [.5])
                edge = edge.repeat(1, 3, 1, 1)
                posterior = self.vae.encode(edge).latent_dist
                edge = torch.cat([posterior.mean, posterior.std], dim=1).cpu().numpy()
        return edge










##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(self.weight.device)
        mu = x.mean(-1, keepdim=True)
        # mu = x.mean(-1)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # sigma = x.var(-1)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


####################################################################################################################################################
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(TransformerBlock, self).__init__()

#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))

#         return x



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 确保 x 是 Tensor，并与 self.attn 处于同一设备
        attn_device = next(self.attn.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=attn_device)

        x = x.to(attn_device)  # 确保数据在正确的设备上

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x












class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)


        self.vae = HEDdetector()
        self.zhuan = nn.Sequential(
            nn.Linear(6272, 3136),
            nn.LayerNorm(3136),
            nn.LeakyReLU(),
            nn.Linear(3136, 1568),
            nn.LayerNorm(1568),
            nn.LeakyReLU(),
            nn.Linear(1568, 784),
            nn.LayerNorm(784),
            nn.LeakyReLU(),
            nn.Linear(784, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
        )
    

        self.attn =  nn.Sequential(*[TransformerBlock(dim=8, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree_LayerNorm') for i in range(8)])



    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)
        
    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            image_features = self.clip.visual(image, control=hiddens)
            
            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)




    # def forward(
    #         self,
    #         image: Optional[torch.Tensor] = None,
    #         text: Optional[torch.Tensor] = None,
    # ):
    #     (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
    #     image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
    #     text_features = self.encode_text(caption, normalize=True) if text is not None else None
    #     text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

    #     return {
    #         "image_features": image_features,
    #         "text_features": text_features,
    #         "image_degra_features": image_degra_features,
    #         "text_degra_features": text_degra_features,
    #         "logit_scale": self.logit_scale.exp()
    #     }


    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
                                 
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None

        text_features = self.vae(image)                                                                                                    # (64, 8, 28, 28)

        text_features = self.attn(text_features).detach().cpu().numpy()+text_features                                                                             # (64, 8, 28, 28)

        
    
        text_features = rearrange(text_features, 'b c h w -> b (c h w)')                                                                   # (64, 6272)
        text_features = torch.from_numpy(text_features).cuda()
        text_features = self.zhuan(text_features)                                                                                          # (64, 512)

        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None                          

        # print('**********************',text_degra_features.shape,'$$$$$$$$$$$$$$$$$$$$$$$$$$$',text_features.shape)                      # (64, 512)
                  
        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }