import torch
import torch.nn as nn


##V1
class DivideNet(nn.Module):
    def __init__(self):
        super(DivideNet, self).__init__()
        # 公共特征提取层 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 分支1：还原模糊图像 (Blurred Branch)
        self.branch_blur = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid()
        )

        # 分支2：还原清晰图像 (Clear Branch)
        self.branch_clear = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.up(self.encoder(x))
        out_blur = self.branch_blur(feat)
        out_clear = self.branch_clear(feat)
        return out_blur, out_clear  # 同时返回两个结果


##V2
# 基础卷积模块
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )


class DivideNet_V2(nn.Module):
    def __init__(self):
        super(DivideNet_V2, self).__init__()

        # --- Encoder (提取特征) ---
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # --- Decoder Clear (提取清晰散斑) ---
        # 重点：利用 Skip Connection 找回锐利边缘
        self.up_c = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_c = conv_block(128, 64)  # 64(up) + 64(skip)
        self.out_c = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Decoder Blur (提取模糊散斑/背景) ---
        self.up_b = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_b = conv_block(128, 64)
        self.out_b = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)  # 保存这个用于 Skip (256x256)
        p1 = self.pool(s1)  # 下采样 (128x128)
        e2 = self.enc2(p1)  # (128x128)

        # Clear Branch
        up_c = self.up_c(e2)  # 回到 256x256
        cat_c = torch.cat([up_c, s1], dim=1)  # 拼接原始高清特征
        out_clear = self.out_c(self.dec_c(cat_c))

        # Blur Branch
        up_b = self.up_b(e2)
        cat_b = torch.cat([up_b, s1], dim=1)
        out_blur = self.out_b(self.dec_b(cat_b))

        return out_clear, out_blur

#V3
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class DivideNet_V3(nn.Module):
    def __init__(self):
        super(DivideNet_V3, self).__init__()
        # Dual Encoders
        self.enc_c = nn.ModuleList([ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128)])
        self.enc_b = nn.ModuleList([ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128)])
        self.pool = nn.MaxPool2d(2)

        # Decoders
        self.up1_c = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_c = ConvBlock(128, 64)
        self.up2_c = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2_c = ConvBlock(64, 32)
        self.out_c = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

        self.up1_b = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_b = ConvBlock(128, 64)
        self.up2_b = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2_b = ConvBlock(64, 32)
        self.out_b = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Clear Path
        c1 = self.enc_c[0](x)
        c2 = self.enc_c[1](self.pool(c1))
        c3 = self.enc_c[2](self.pool(c2))
        # Blur Path
        b1 = self.enc_b[0](x)
        b2 = self.enc_b[1](self.pool(b1))
        b3 = self.enc_b[2](self.pool(b2))

        # Decoding
        pc = self.out_c(self.dec2_c(torch.cat([self.up2_c(self.dec1_c(torch.cat([self.up1_c(c3), c2], 1))), c1], 1)))
        pb = self.out_b(self.dec2_b(torch.cat([self.up2_b(self.dec1_b(torch.cat([self.up1_b(b3), b2], 1))), b1], 1)))
        return pc, pb


import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class DeblurUNet(nn.Module):
    def __init__(self):
        super(DeblurUNet, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 保存输入用于最后的全局残差 (可选)
        identity = x

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        # Decoder with Skip Connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)