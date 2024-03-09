import numpy as np
from segmentation_models_pytorch.encoders import get_encoder
import torch
from torch import nn
from segment_anything import sam_encoder_model_registry
from importlib import import_module
from sam_lora_bias import LoRA_bias_Sam
from segmentation_models_pytorch.unet.decoder import DecoderBlock
from segmentation_models_pytorch.base import SegmentationHead, modules
from typing import Optional, Union, List
from fusion_module import Fusion


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = modules.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = modules.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            skip=True
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = 256
        in_channels = [head_channels] + list(decoder_channels[:-1])
        if skip:
            skip_channels = list(encoder_channels[1:]) + [0]
        else:
            skip_channels = [0] * len(encoder_channels)
        self.skip = skip
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[0]
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            if self.skip:
                x = decoder_block(x, skip)
            else:
                x = decoder_block(x)

        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, dw=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        if dw:
            self.conv = DepthwiseSeparableConv(inp_dim, out_dim, kernel_size, stride, bias=bias)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class SAM_UNet(nn.Module):
    def __init__(self, num_classes,
                 encoder_name: str = "mobilenet_v2",
                 image_size: int = 512,
                 encoder_depth: int = 4,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 activation: Optional[Union[str, callable]] = None, ):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_encoder = get_encoder(name=encoder_name, in_channels=3, depth=encoder_depth, weights=encoder_weights)
        sam_encoder, img_embedding_size = sam_encoder_model_registry['vit_b'](image_size=image_size,
                                                                              num_classes=num_classes-1,
                                                                              checkpoint='checkpoints/sam_encoder_vit_b_01ec64.pth',
                                                                              pixel_mean=[0, 0, 0],
                                                                              pixel_std=[1, 1, 1])
        self.img_embedding_size = img_embedding_size

        pkg = import_module('sam_lora_bias')
        self.sam_encoder = pkg.LoRA_bias_Sam(sam_encoder, 4)

        self.decoder = UnetDecoder(
            encoder_channels=self.cnn_encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=4,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )

        self.fusion = Fusion(self.cnn_encoder.out_channels[-1], 256)


    def forward(self, x):
        cnn_feature = self.cnn_encoder(x)
        sam_feature = self.sam_encoder(x)
        cnn_feature[-1] = self.fusion(cnn_feature[-1], sam_feature)
        output = self.segmentation_head(self.decoder(cnn_feature))
        return output

