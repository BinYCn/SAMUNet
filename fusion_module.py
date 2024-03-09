import torch
from torch import nn
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


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


class DepthwisedilateConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, dw=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        if dw:
            self.conv = DepthwiseSeparableConv(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
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


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, dw):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False, dw=dw)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False, dw=dw)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False, dw=dw)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False, dw=dw)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Foucs_ECA_Dia135(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DepthwisedilateConv(in_channels, in_channels // 2, 3, 1, 1, dilation=1)
        self.conv2 = DepthwisedilateConv(in_channels, in_channels // 4, 3, 1, 3, dilation=3)
        self.conv3 = DepthwisedilateConv(in_channels, in_channels // 4, 3, 1, 5, dilation=5)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.gap(y)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Foucs_SPA_Dia135(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DepthwisedilateConv(in_channels, in_channels // 2, 3, 1, 1, dilation=1)
        self.conv2 = DepthwisedilateConv(in_channels, in_channels // 4, 3, 1, 3, dilation=3)
        self.conv3 = DepthwisedilateConv(in_channels, in_channels // 4, 3, 1, 5, dilation=5)
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, 1, bn=True, relu=False, bias=False, dw=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.compress(y)
        y = self.spatial(y)
        return x * y.expand_as(x)


class Fusion(nn.Module):
    def __init__(self, cnn_channel, sam_channel):
        super().__init__()

        self.cnn_att = Foucs_SPA_Dia135(cnn_channel)
        self.sam_att = Foucs_ECA_Dia135(sam_channel)

        self.W_cnn = Conv(cnn_channel, sam_channel, 1, bn=True, relu=False, dw=True)
        self.W_sam = Conv(sam_channel, sam_channel, 1, bn=True, relu=False, dw=True)
        self.W = Conv(sam_channel, sam_channel, 3, bn=True, relu=True, dw=True)

        self.concatconv = Conv(cnn_channel + sam_channel + sam_channel, sam_channel, 3, bn=True, dw=True)

    def forward(self, cnn_feature, sam_feature):
        W_cnn = self.W_cnn(cnn_feature)
        W_sam = self.W_sam(sam_feature)
        dot = self.W(W_cnn * W_sam)

        cnn_feature = self.cnn_att(cnn_feature)
        sam_feature = self.sam_att(sam_feature)
        fusion_feature = self.concatconv(torch.cat([cnn_feature, sam_feature, dot], dim=1))
        return fusion_feature
