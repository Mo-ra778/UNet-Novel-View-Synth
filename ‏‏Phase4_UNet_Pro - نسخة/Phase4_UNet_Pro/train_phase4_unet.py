"""
Phase 4 Training Script - Full Coverage Edition 🚀
التدريب الاحترافي: U-Net + Perceptual Loss + On-the-Fly Generation
متضمن تحسينات السرعة (FP16) وإدارة الذاكرة

✨ الجديد في هذه النسخة:
- تغطية كاملة: Elevation 0°-90° | Azimuth 0°-360°
- زوايا عشوائية جديدة في كل epoch
- توزيع منظم (Stratified Random)
- تعميم قوي على جميع الزوايا
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os
from pathlib import Path
import time
from datetime import timedelta
import gc
import numpy as np  # ✅ تم إضافة هذا السطر
from tqdm import tqdm # ✅ إضافة شريط التقدم

from model_unet import get_unet_model
from loss_perceptual import CombinedLoss
from dataloader_phase4 import create_phase4_dataloaders

# ============== إعدادات التدريب المحسنة ==============
TOTAL_EPOCHS = 30      # عدد الدورات الكاملة على البيانات (150,000 عينة * 20)
# إجمالي الخطوات التقريبي: (150,000 / 32) * 20 ≈ 93,750 خطوة
BATCH_SIZE = 32              # زيادة الحجم لاستغلال كفاءة كرت الشاشة
NUM_WORKERS = 6             # ⚠️ تقليل العمال لتوفير الذاكرة على Windows
LEARNING_RATE = 2e-4         # زيادة طفيفة لمواكبة الباتش الأكبر
VAL_FREQ_STEPS = 100         # تحديث اللوحة الحية كل 100 خطوة ⚡
SAVE_FREQ_EPOCHS = 1         # حفظ نسخة كل دورة (Epoch)
LOG_FREQ = 50                # طباعة في التيرمينال كل 50 خطوة 
OUTPUT_DIR = 'results_phase4_optimized'
# =====================================================


class Phase4Trainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # تنظيف الذاكرة في البداية
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        print("=" * 70)
        print("🚀 Phase 4 Ultimate Training System (Optimized)")
        print("=" * 70)
        print(f"📊 Configuration:")
        print(f"   - Device: {self.device} (Mixed Precision Enabled ⚡)")
        print(f"   - Total Epochs: {TOTAL_EPOCHS}")
        print(f"   - Batch Size: {BATCH_SIZE}")
        print(f"   - Workers: {NUM_WORKERS}")
        print(f"   - Output: {OUTPUT_DIR}/")
        print("=" * 70)
        
        # 1. Model
        print("\n📦 Loading U-Net Model...")
        self.model = get_unet_model(self.device)
        
        # 2. Loss
        print("📦 Loading Loss Function (L1 + Perceptual)...")
        self.criterion = CombinedLoss(lambda_perceptual=0.2, device=self.device)
        
        # 3. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=1e-4
        )
        
        # 4. DataLoaders - سنعيد إنشاءها في كل epoch
        print("📦 Creating Optimized DataLoaders...")
        # ملاحظة: سيتم إعادة إنشاء train_loader في كل epoch بزوايا جديدة
        self.train_loader, self.val_loader = self._create_dataloaders(epoch_seed=0)
        
        # حساب إجمالي الخطوات تلقائياً
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = TOTAL_EPOCHS * self.steps_per_epoch
        print(f"   ℹ️  Steps per Epoch: {self.steps_per_epoch}")
        print(f"   ℹ️  Total Training Steps: {self.total_steps}")

        # 5. Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=LEARNING_RATE * 10,
            total_steps=self.total_steps,
            pct_start=0.1
        )
        
        # 6. FP16 Scaler
        self.scaler = GradScaler()
        
        self.history = {'step': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        
        # محاولة استعادة التدريب السابق
        self.start_epoch = 1
        self.global_step = 0
        self.best_loss = float('inf')
        self._resume_training()
        
        print("\n✅ Initialization Complete!")
    
    def _create_dataloaders(self, epoch_seed=None):
        """
        إنشاء DataLoaders مع seed محدد
        يتم استدعاءها في كل epoch بـ seed مختلف للحصول على زوايا جديدة
        """
        return create_phase4_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pairs_per_shape=10,
            epoch_seed=epoch_seed
        )

    def _resume_training(self):
        """البحث عن آخر checkpoint واستكمال التدريب"""
        checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return
        
        # استخراج أحدث ملف بناءً على رقم الـ epoch
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_ckpt = os.path.join(OUTPUT_DIR, checkpoints[-1])
        
        print(f"\n🔄 Found checkpoint: {latest_ckpt}")
        print("   Loading training state...")
        
        checkpoint = torch.load(latest_ckpt, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"   ✅ Resuming from Epoch {self.start_epoch}, Step {self.global_step}")

    def train(self):
        print(f"\n🔥 Starting Training...")
        self.model.train()
        global_start = time.time()
        
        step = self.global_step
        best_loss = self.best_loss
        running_loss = 0.0
        current_img_per_sec = 0.0
        current_psnr = 0.0
        
        # شريط التقدم الكلي للـ Epochs
        epoch_pbar = tqdm(range(self.start_epoch, TOTAL_EPOCHS + 1), desc="Training Epochs", position=0)
        
        for epoch in epoch_pbar:
            # 🧹 تنظيف الذاكرة قبل إنشاء DataLoader جديد!
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                del self.train_loader
            gc.collect()
            torch.cuda.empty_cache()
            
            # 🎲 إعادة إنشاء DataLoader بزوايا جديدة!
            epoch_seed = 1000 + epoch  # seed مختلف لكل epoch
            self.train_loader, _ = self._create_dataloaders(epoch_seed=epoch_seed)
            tqdm.write(f"\n🎯 Epoch {epoch}: Generated new random views (seed={epoch_seed})")
            
            # شريط التقدم للـ Steps داخل كل Epoch
            step_pbar = tqdm(enumerate(self.train_loader), 
                             total=len(self.train_loader), 
                             desc=f"Epoch {epoch}", 
                             position=1, 
                             leave=False)
            
            for i, batch in step_pbar:
                step = self.global_step + 1
                self.global_step = step
                
                # === Training Step (Optimized) ===
                # نقل البيانات للذاكرة (Non-blocking)
                source_img = batch['source_image'].to(self.device, non_blocking=True)
                # source_cam = تم إزالته! 🎯
                target_img = batch['target_image'].to(self.device, non_blocking=True)
                target_cam = batch['target_camera'].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True) # يوفر الذاكرة أكثر من zero_grad()
                
                # Mixed Precision Context
                with autocast():
                    pred_img = self.model(source_img, target_cam)  # 🎯 فقط target_cam!
                    loss = self.criterion(pred_img, target_img)
                
                # Backward with Scaler
                self.scaler.scale(loss).backward()
                
                # Gradient Clipping (Unscale first)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                running_loss += loss.item()
                
                # === Logging ===
                if step % LOG_FREQ == 0:
                    avg_loss = running_loss / LOG_FREQ
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    elapsed = time.time() - global_start
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    current_img_per_sec = steps_per_sec * BATCH_SIZE
                    
                    # تحديث شريط التقدم بمعلومات لحظية
                    step_pbar.set_postfix({
                        "Loss": f"{avg_loss:.4f}",
                        "LR": f"{current_lr:.6f}",
                        "Speed": f"{current_img_per_sec:.0f} img/s",
                        "PSNR": f"{current_psnr:.2f} dB"
                    })
                    
                    running_loss = 0.0
                
                # === Validation & Dashboard ===
                if step % VAL_FREQ_STEPS == 0:
                    val_loss, val_psnr, val_ssim = self.validate(step)
                    current_psnr = val_psnr
                    
                    self.history['step'].append(step)
                    self.history['train_loss'].append(loss.item())
                    self.history['val_loss'].append(val_loss)
                    self.history['lr'].append(current_lr)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(self.model.state_dict(), 
                                  os.path.join(OUTPUT_DIR, 'best_model.pth'))
                        tqdm.write(f"   ✅ Best Model Saved! Loss: {best_loss:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")
                    
                    width=0.4
                    
                    # تحديث لوحة المراقبة الحية (بكل التفاصيل)
                    self.update_live_dashboard(
                        epoch, TOTAL_EPOCHS, 
                        step, self.total_steps, 
                        BATCH_SIZE, 
                        batch, current_img_per_sec, current_psnr, val_ssim,
                        loss.item(), val_loss, best_loss, current_lr
                    )
                    self.model.train()
                    
                    # تنظيف الذاكرة بعد التحقق
                    torch.cuda.empty_cache()
            
            # === End of Epoch Save ===
            ckpt_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': best_loss,
                'history': self.history
            }, ckpt_path)
            print(f"   💾 Checkpoint saved: {ckpt_path}")
        
        # === Final Save ===
        total_time = str(timedelta(seconds=int(time.time() - global_start)))
        print("\n" + "=" * 70)
        print("🎉 Training Complete!")
        print(f"⏱️  Total Time: {total_time}")
        print(f"📉 Best Validation Loss: {best_loss:.4f}")
        
        torch.save(self.model.state_dict(), 
                  os.path.join(OUTPUT_DIR, 'final_model.pth'))
        print("=" * 70)

    def validate(self, step):
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        num_batches = 5
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_batches:
                    break
                
                source_img = batch['source_image'].to(self.device, non_blocking=True)
                # source_cam = تم إزالته! 🎯
                target_img = batch['target_image'].to(self.device, non_blocking=True)
                target_cam = batch['target_camera'].to(self.device, non_blocking=True)
                
                with autocast():
                    pred_img = self.model(source_img, target_cam) # 🎯 فقط target_cam!
                    loss = self.criterion(pred_img, target_img)
                
                val_loss += loss.item()
                
                # حساب PSNR
                mse = torch.mean((pred_img - target_img) ** 2)
                psnr = -10 * torch.log10(mse + 1e-8)
                val_psnr += psnr.item()
                
                # حساب SSIM تقريبي (بسيط لغرض العرض السريع)
                val_ssim += 1.0 - torch.mean(torch.abs(pred_img - target_img)).item() # بديل سريع للـ SSIM
        
        avg_loss = val_loss / num_batches
        avg_psnr = val_psnr / num_batches
        avg_ssim = val_ssim / num_batches
        
        return avg_loss, avg_psnr, avg_ssim

    def update_live_dashboard(self, epoch, total_epochs, step, total_steps, batch_size, batch, speed, psnr, ssim, train_loss, val_loss, best_loss, lr):
        """إنشاء لوحة تحكم حية شاملة"""
        
        # 1. تجهيز الصور
        self.model.eval()
        with torch.no_grad():
            src = batch['source_image'].to(self.device)
            # src_cam = تم إزالته! 🎯
            tgt = batch['target_image'].to(self.device)
            tgt_cam = batch['target_camera'].to(self.device)
            
            with autocast():
                pred = self.model(src, tgt_cam)  # 🎯 فقط target_cam!
        
        # 2. إعداد اللوحة (GridSpec) - تصميم جديد يعطي مساحة أكبر للصور
        fig = plt.figure(figsize=(16, 14)) # زيادة الحجم الكلي
        
        # عنوان مفصل
        title_text = (
            f"Phase 4 Pro Dashboard 🚀\n"
            f"Epoch: {epoch}/{total_epochs}  |  Step: {step}/{total_steps}  |  Batch: {batch_size}\n"
            f"Speed: {speed:.1f} img/s  |  PSNR: {psnr:.2f} dB  |  Sim: {ssim:.4f}"
        )
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
        
        # تخطيط جديد: 4 صفوف (صف المخططات + 3 صفوف صور)
        # تخصيص ارتفاعات الصفوف: المخططات تأخذ مساحة صغيرة، والصور تأخذ الباقي
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.5, 1.5, 1.5], hspace=0.3)
        
        # === المخططات (الصف الأول) ===
        ax_loss = fig.add_subplot(gs[0, 0]) # Loss
        ax_stats = fig.add_subplot(gs[0, 1:]) # Stats Box (كبير وواضح)
        
        # رسم Loss
        if len(self.history['step']) > 0:
            ax_loss.plot(self.history['step'], self.history['train_loss'], label='Train', linewidth=1.5, alpha=0.7)
            ax_loss.plot(self.history['step'], self.history['val_loss'], label='Val', linewidth=2, color='red')
            ax_loss.set_title('Loss History', fontsize=12)
            ax_loss.legend(prop={'size': 10})
            ax_loss.grid(True, alpha=0.3)
            
        # مربع الإحصائيات (Stats Box)
        ax_stats.axis('off')
        stats_text = (
            f"📊 Training Statistics:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Current Train Loss: {train_loss:.5f}\n"
            f"• Current Val Loss:   {val_loss:.5f}\n"
            f"• Best Val Loss:      {best_loss:.5f}\n"
            f"• Learning Rate:      {lr:.7f}\n"
            f"• Avg PSNR:           {psnr:.2f} dB\n"
            f"• Avg Similarity:     {ssim:.4f}\n"
        )
        ax_stats.text(0.05, 0.5, stats_text, fontsize=16, va='center', fontfamily='monospace')
        
        # === الصور (صفوف للدات) ===
        # نعرض 3 عينات فقط ولكن بحجم كبير 
        samples_to_show = min(src.size(0), 3)
        cols = ['Source Input', 'Prediction', 'Ground Truth']
        
        for i in range(samples_to_show):
            row = i + 1 # نبدأ من الصف الثاني
            
            # 1. Source Image
            ax1 = fig.add_subplot(gs[row, 0])
            img_src = src[i].float().permute(1, 2, 0).cpu().numpy()
            ax1.imshow(img_src)
            ax1.axis('off')
            if i == 0: ax1.set_title(cols[0], fontsize=14, fontweight='bold', color='#1f77b4')
            
            # 2. Prediction Image
            ax2 = fig.add_subplot(gs[row, 1])
            img_pred = pred[i].float().permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ax2.imshow(img_pred)
            ax2.axis('off')
            if i == 0: ax2.set_title(cols[1], fontsize=14, fontweight='bold', color='#2ca02c')
            
            # 3. Target Image
            ax3 = fig.add_subplot(gs[row, 2])
            img_tgt = tgt[i].float().permute(1, 2, 0).cpu().numpy()
            ax3.imshow(img_tgt)
            ax3.axis('off')
            if i == 0: ax3.set_title(cols[2], fontsize=14, fontweight='bold', color='black')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # ترك مساحة للعناوين
        
        # حفظ اللوحة الحية (يتم استبدالها كل 100 خطوة)
        live_path = os.path.join(OUTPUT_DIR, 'live_dashboard.png')
        plt.savefig(live_path, dpi=100)
        
        # حفظ نسخة للأرشيف كل 500 خطوة فقط
        if step % 500 == 0:
            history_path = os.path.join(OUTPUT_DIR, f'dashboard_step_{step}.jpg')
            plt.savefig(history_path, dpi=80) 
        
        plt.close()


if __name__ == "__main__":
    trainer = Phase4Trainer()
    trainer.train()
