from functools import partial

import torch
import torch.nn as nn

import torchaudio

import soundfile as sf

from timm.models.layers import to_2tuple, trunc_normal_

from utils.stft import STFT, LogmelFilterBank
from utils.vision_transformer import VisionTransformer as _VisionTransformer

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

import importlib
import os
import numpy as np

def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


class SpatialAST(_VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_cls_tokens=3, **kwargs):
        super().__init__(**kwargs)
        img_size = (1024, 128) # 1024, 128
        in_chans = 1
        emb_dim = 768

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens, std=.02)


        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=(16,16), 
            in_chans=in_chans, embed_dim=emb_dim, stride=16 # might need to change this to 4? not sure yet
        ) # no overlap. stride=img_size=16
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=True
        )
        
        import librosa
        self.melW = librosa.filters.mel(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, fmax=14000
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, 
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )
        
        # self.conv_downsample = nn.Sequential(
        #     conv3x3(4, 1), 
        #     nn.BatchNorm2d(1),
        #     nn.GELU(),
        # )

        # Changed this to sample down to 4 channels
        self.conv_downsample = nn.Sequential(
            conv3x3(16, 1), 
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.bn = nn.BatchNorm2d(4, affine=False) # changed this to 4 channels
        del self.norm  # remove the original norm

        self.target_frame = 1024

        self.dis_norm = kwargs['norm_layer'](emb_dim)
        self.doa_norm = kwargs['norm_layer'](emb_dim)
        self.fc_norm = kwargs['norm_layer'](emb_dim)

        self.distance_head = nn.Linear(emb_dim, 21 * 2) # [0:10:0.5], 21 classes, Account for up to 2 active sound locations 
        self.azimuth_head = nn.Linear(emb_dim, 360 * 2) # Account for up to 2 active sound locations
        self.elevation_head = nn.Linear(emb_dim, 180 * 2) # Account for up to 2 active sound locations

        trunc_normal_(self.head.weight, std=2e-5)
        trunc_normal_(self.distance_head.weight, std=2e-5)
        trunc_normal_(self.azimuth_head.weight, std=2e-5)
        trunc_normal_(self.elevation_head.weight, std=2e-5)

        model_config = {"model": {
                        "module": "cdbpnproj",
                        "main": "CDBPNProj",
                        "args": {}
        }}

        #model_checkpoint_path = "/scratch/data/repos/LAM/train_all_spatial_scaper_4ch_wideband/checkpoints/model_0003.pth"
        ##model_checkpoint_path = "/scratch/data/repos/LAM/kitchensink_eval_locata/checkpoints/model_0012.pth"
        #device = 'cuda:0'
        #self.lam_model = initialize_config(model_config['model'])
        #self.lam_model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
        #self.lam_model.to(device)

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        N, L, D = x.shape  # batch, length, dim
        T, F = 64, 8
        
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0, 2, 1, 3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #bsz, 512, 768 (unmasked)

        x = x + self.pos_embed[:, 1:, :]

        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)

        cls_tokens = self.cls_tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # bsz, 512 + 2 + 10, 768 
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        return x

    
    # overwrite original timm
    # need to edit this
    def forward(self, waveforms, reverbs, mask_t_prob=0.0, mask_f_prob=0.0):
        #waveforms = torchaudio.functional.fftconvolve(waveforms, reverbs*0.1258, mode='full')[..., :waveforms.shape[-1]]
        # Call LAM encoder
        #lam_tokens = self.lam_model(waveforms)
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        
        real, imag = self.spectrogram_extractor(waveforms) 

        log_mel = self.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)
        log_mel = self.bn(log_mel)
        
        # Compute IPD between channel pairs (1 vs 2, and 3 vs 4)
        IPD_12 = torch.atan2(imag[1::4], real[1::4]) - torch.atan2(imag[0::4], real[0::4])  # mic 1 vs mic 2
        IPD_13 = torch.atan2(imag[2::4], real[2::4]) - torch.atan2(imag[0::4], real[0::4])  # mic 1 vs mic 3
        IPD_14 = torch.atan2(imag[3::4], real[3::4]) - torch.atan2(imag[0::4], real[0::4])  # mic 1 vs mic 4
        IPD_23 = torch.atan2(imag[2::4], real[2::4]) - torch.atan2(imag[1::4], real[1::4])  # mic 2 vs mic 3
        IPD_24 = torch.atan2(imag[3::4], real[3::4]) - torch.atan2(imag[1::4], real[1::4])  # mic 2 vs mic 4
        IPD_34 = torch.atan2(imag[3::4], real[3::4]) - torch.atan2(imag[2::4], real[2::4])  # mic 3 vs mic 4
        # Concatenate the IPD results along the channel dimension
        IPD = torch.cat([IPD_12, IPD_13, IPD_14, IPD_23, IPD_24, IPD_34], dim=1)
        x = torch.cat([log_mel, torch.matmul(torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1), self.logmel_extractor.melW)], dim=1)
        # x = log_mel

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True) # might need to modify this


        x = self.conv_downsample(x)
        if self.training:
            x = x.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        x = self.patch_embed(x)
        x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)

        dis_token = x[:, 0]
        doa_token = x[:, 1]
        cls_tokens = x[:, 2]

        dis_token = self.dis_norm(dis_token)
        doa_token = self.doa_norm(doa_token)
        cls_tokens = self.fc_norm(cls_tokens)

        classifier = self.head(cls_tokens)
        distance = self.distance_head(dis_token)
        azimuth = self.azimuth_head(doa_token)
        elevation = self.elevation_head(doa_token)

        return classifier, distance, azimuth, elevation


def build_AST(**kwargs):
    model = SpatialAST(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
