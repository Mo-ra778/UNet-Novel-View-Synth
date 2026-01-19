"""
Perceptual Loss (VGG-Based) 🧠👁️
دالة خسارة متقدمة تستخدم شبكة VGG لمقارنة "ملامح" الصورة وليس فقط البكسلات.
هذا هو السر للحصول على صور حادة (Sharp) بدلاً من الضبابية.
"""

import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        print("⏳ Loading VGG19 model for perceptual loss...")
        
        # تحميل VGG19 المدرب مسبقاً
        # نستخدم weights='DEFAULT' لتحميل أفضل أوزان متوفرة
        try:
            vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        except:
            # Fallback for older torch versions
            vgg_pretrained_features = models.vgg19(pretrained=True).features
            
        # نأخذ طبقات معينة لاستخراج الملامح (Features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(2): # Relu1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7): # Relu2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12): # Relu3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21): # Relu4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        # تجميد الأوزان (لا نريد تدريب VGG، فقط استخدامه)
        self.eval()

    def forward(self, pred, target):
        # نفترض الصور 0-1. VGG يتوقع تطبيعاً معيناً لكن التجربة أثبتت أنه يعمل جيداً بدونه للمقارنة النسبية.
        
        h_relu1_pred = self.slice1(pred)
        h_relu1_target = self.slice1(target)
        
        h_relu2_pred = self.slice2(h_relu1_pred)
        h_relu2_target = self.slice2(h_relu1_target)
        
        h_relu3_pred = self.slice3(h_relu2_pred)
        h_relu3_target = self.slice3(h_relu2_target)
        
        h_relu4_pred = self.slice4(h_relu3_pred)
        h_relu4_target = self.slice4(h_relu3_target)
        
        # حساب الفرق بين الملامح (L1 Loss)
        loss = torch.nn.functional.l1_loss(h_relu1_pred, h_relu1_target) + \
               torch.nn.functional.l1_loss(h_relu2_pred, h_relu2_target) + \
               torch.nn.functional.l1_loss(h_relu3_pred, h_relu3_target) + \
               torch.nn.functional.l1_loss(h_relu4_pred, h_relu4_target)
               
        return loss / 4

class CombinedLoss(nn.Module):
    """
    الخسارة المركبة:
    Loss = L1_Loss + (Lambda * Perceptual_Loss)
    L1 تضبط الألوان، Perceptual تضبط التفاصيل
    """
    def __init__(self, lambda_perceptual=0.2, device='cuda'):
        super().__init__()
        self.l1 = nn.L1Loss()
        
        try:
            self.perceptual = VGGPerceptualLoss().to(device)
            self.use_perceptual = True
            print("✅ VGG Perceptual Loss initialized successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load VGG ({e}). Using only L1 Loss.")
            self.use_perceptual = False
            
        self.lambda_p = lambda_perceptual

    def forward(self, pred, target):
        # 1. Pixel Loss (L1) - للحفاظ على الألوان والهيكل العام
        loss_l1 = self.l1(pred, target)
        
        loss_p = 0.0
        # 2. Perceptual Loss - للحفاظ على التفاصيل والحدة
        if self.use_perceptual:
            loss_p = self.perceptual(pred, target)
            
        # Total Loss
        return loss_l1 + (self.lambda_p * loss_p)
