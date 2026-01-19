"""
U-Net Architecture with Skip Connections 🏗️
موديل احترافي يستخدم U-Net للحفاظ على التفاصيل (Skip Connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3DModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet3DModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Camera Embedding Layers
        # 🎯 تحسين: نستخدم فقط كاميرا الهدف (3 قيم)
        # النموذج يستنتج موقع المصدر من الصورة نفسها!
        self.cam_mlp = nn.Sequential(
            nn.Linear(3, 128),  # 3 قيم فقط: target camera
            nn.ReLU(),
            nn.Linear(128, 1024 // factor),
            nn.ReLU()
        )

        # Decoder
        # up1 input: x5(512) + x4(512) = 1024 channels -> out: 256 // 2 = 128
        self.up1 = Up(1024, 512 // factor, bilinear)
        
        # up2 input: up1_out(128) + x3(256) = 384 channels -> out: 128 // 2 = 64
        self.up2 = Up(384, 256 // factor, bilinear)
        
        # up3 input: up2_out(64) + x2(128) = 192 channels -> out: 64 // 2 = 32
        self.up3 = Up(192, 128 // factor, bilinear)
        
        # up4 input: up3_out(32) + x1(64) = 96 channels -> out: 64 // 2 = 32
        self.up4 = Up(96, 64, bilinear)
        
        self.outc = OutConv(32, n_classes)

    def forward(self, x, tgt_cam):
        """
        🎯 التحسين: مدخل واحد فقط للكاميرا!
        
        Args:
            x: صورة المصدر [B, 3, H, W]
            tgt_cam: موقع الكاميرا المطلوب [B, 3] (elevation, azimuth, distance)
        
        Returns:
            صورة الهدف [B, 3, H, W]
        
        ملاحظة: النموذج يستنتج موقع كاميرا المصدر من الصورة نفسها! 🧠
        """
        # 1. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 2. Camera Injection (Bottleneck)
        # دمج كاميرا الهدف مع الـ features
        # x5 shape: [B, 512, H/16, W/16]
        
        # تحضير كاميرا الهدف فقط
        cam_emb = self.cam_mlp(tgt_cam) # [B, 512]
            
        # توسعة الكاميرا لتناسب الصورة
        cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).expand_as(x5)
            
        # دمج بالجمع (أو الكونكات) - هنا سنستخدم الجمع البسيط للسرعة والفعالية في الـ conditional gen
        x5 = x5 + cam_emb

        # 3. Decoder with Skip Connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits) # لضمان الإخراج بين 0 و 1

def get_unet_model(device='cuda'):
    model = UNet3DModel().to(device)
    print(f"🏗️ U-Net Model initialized on {device}")
    return model
