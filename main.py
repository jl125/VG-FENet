import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_fft=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.use_fft = use_fft
        if self.use_fft:
            self.fft_process = FFT_Process(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_fft:
            x = self.fft_process(x)
        return x


class FFT_Process(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)

    def _make_process_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out + x


class Encoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels, use_fft=True)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, use_fft=True)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4


class Decoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 8, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, e1, e2, e3, e4, v_enhanced_feat=None):
        if v_enhanced_feat is not None:
            e4 = e4 + v_enhanced_feat

        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out_conv(d2)


class RetinexBrightnessSubNet(nn.Module):
    def __init__(self, base_channels=32, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.decompose_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_channels, 2, 3, padding=1)
        )
        self.illumination_conv = nn.Sequential(
            nn.Conv2d(1, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.reflectance_conv = nn.Sequential(
            nn.Conv2d(1, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, v):
        v_log = torch.log(v + self.eps)
        lr_log = self.decompose_conv(v)
        l_log, r_log = lr_log[:, 0:1, :, :], lr_log[:, 1:2, :, :]
        l = torch.exp(l_log)
        r = torch.exp(r_log)

        l_optimized = self.illumination_conv(l)
        r_optimized = self.reflectance_conv(r)

        v_enhanced = l_optimized * r_optimized
        return v_enhanced, l_optimized, r_optimized


def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    max_rgb = torch.max(rgb, dim=1, keepdim=True)[0]
    min_rgb = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = max_rgb - min_rgb + 1e-8
    h = torch.zeros_like(max_rgb)
    mask = delta > 1e-5
    r_mask = (max_rgb == r) & mask
    g_mask = (max_rgb == g) & mask
    b_mask = (max_rgb == b) & mask
    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    h = h / 6.0
    s = torch.where(max_rgb > 1e-3, delta / max_rgb, torch.zeros_like(max_rgb))
    v = max_rgb
    return torch.cat([h, s, v], dim=1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    h = h * 360.0
    c = v * s
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
    m = v - c
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    r[(h >= 0) & (h < 60)] = c[(h >= 0) & (h < 60)]
    g[(h >= 0) & (h < 60)] = x[(h >= 0) & (h < 60)]
    r[(h >= 60) & (h < 120)] = x[(h >= 60) & (h < 120)]
    g[(h >= 60) & (h < 120)] = c[(h >= 60) & (h < 120)]
    g[(h >= 120) & (h < 180)] = c[(h >= 120) & (h < 180)]
    b[(h >= 120) & (h < 180)] = x[(h >= 120) & (h < 180)]
    g[(h >= 180) & (h < 240)] = x[(h >= 180) & (h < 240)]
    b[(h >= 180) & (h < 240)] = c[(h >= 180) & (h < 240)]
    r[(h >= 240) & (h < 300)] = x[(h >= 240) & (h < 300)]
    b[(h >= 240) & (h < 300)] = c[(h >= 240) & (h < 300)]
    r[(h >= 300) & (h < 360)] = c[(h >= 300) & (h < 360)]
    b[(h >= 300) & (h < 360)] = x[(h >= 300) & (h < 360)]
    r += m
    g += m
    b += m
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)


class RetinexUNetLLIE(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.encoder = Encoder(base_channels)
        self.decoder = Decoder(base_channels)
        self.brightness_subnet = RetinexBrightnessSubNet(base_channels)
        self.v_feat_conv = nn.Conv2d(1, base_channels * 8, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)

        hsv = rgb_to_hsv(x)
        v = hsv[:, 2:3, :, :]
        v_enhanced, l_optimized, r_optimized = self.brightness_subnet(v)

        v_feat = self.v_feat_conv(v_enhanced)
        v_feat_down = F.interpolate(v_feat, size=e4.shape[2:], mode='bilinear')

        final_out = self.decoder(e1, e2, e3, e4, v_feat_down) + x

        hsv_enhanced = hsv.clone()
        hsv_enhanced[:, 2:3, :, :] = v_enhanced
        brightness_out = hsv_to_rgb(hsv_enhanced)

        return final_out, brightness_out, l_optimized, r_optimized