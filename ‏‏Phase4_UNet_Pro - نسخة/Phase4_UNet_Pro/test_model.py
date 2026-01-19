"""
🧪 Test Script for Phase 4 U-Net Model
اختبار النموذج المدرب على أشكال جديدة

يقوم هذا السكربت بـ:
1. تحميل النموذج المدرب
2. توليد أشكال عشوائية جديدة
3. اختبار قدرة النموذج على التنبؤ بالزوايا المختلفة
4. حفظ صور المقارنة
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

# استيراد الوحدات المحلية
from model_unet import get_unet_model
from dataloader_phase4 import FixedShapeGenerator, generate_fibonacci_views, render_shape

# ============== الإعدادات ==============
MODEL_PATH = 'results_phase4_optimized/best_model.pth'
OUTPUT_DIR = 'test_results'
NUM_TEST_SAMPLES = 10  # عدد العينات للاختبار
IMAGE_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ======================================

def load_model():
    """تحميل النموذج المدرب"""
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = get_unet_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded successfully on {DEVICE}")
    return model

def generate_test_sample(shape_id):
    """توليد عينة اختبار (شكل جديد لم يراه الموديل)"""
    generator = FixedShapeGenerator()
    views = generate_fibonacci_views(60)
    
    # توليد الشكل
    shape_data = generator.generate_shape_by_id(shape_id)
    shape_color = generator.get_shape_color(shape_id)
    
    # اختيار زاويتين عشوائيتين
    np.random.seed(shape_id * 7 + 999)
    src_idx, tgt_idx = np.random.choice(60, 2, replace=False)
    
    src_view = views[src_idx]
    tgt_view = views[tgt_idx]
    
    # رسم الصور (render_shape تُعيد tensor مباشرة!)
    src_tensor = render_shape(shape_data, src_view[0], src_view[1], shape_color)
    tgt_tensor = render_shape(shape_data, tgt_view[0], tgt_view[1], shape_color)
    
    # إحداثيات الكاميرا (نفس التنسيق في التدريب - 3 قيم)
    src_cam = torch.tensor([
        src_view[0] / 90.0,
        src_view[1] / 180.0 - 1.0,
        1.0
    ], dtype=torch.float32)
    
    tgt_cam = torch.tensor([
        tgt_view[0] / 90.0,
        tgt_view[1] / 180.0 - 1.0,
        1.0
    ], dtype=torch.float32)
    
    return {
        'source_image': src_tensor,
        'target_image': tgt_tensor,
        'source_camera': src_cam,
        'target_camera': tgt_cam,
        'shape_id': shape_id,
        'src_view': src_view,
        'tgt_view': tgt_view
    }

def calculate_metrics(pred, target):
    """حساب مقاييس الجودة"""
    # MSE
    mse = torch.mean((pred - target) ** 2).item()
    
    # PSNR
    psnr = -10 * np.log10(mse + 1e-8)
    
    # Similarity (1 - MAE)
    sim = 1.0 - torch.mean(torch.abs(pred - target)).item()
    
    return {'mse': mse, 'psnr': psnr, 'similarity': sim}

def test_model(model, num_samples=10):
    """اختبار النموذج على عدة عينات"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n🧪 Testing model on {num_samples} new shapes...")
    print("=" * 60)
    
    all_metrics = []
    
    # استخدام أشكال جديدة (IDs أكبر من 15000 للتأكد أنها لم تُستخدم في التدريب)
    test_shape_ids = range(20000, 20000 + num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i, shape_id in enumerate(test_shape_ids):
        print(f"\n📐 Testing Shape #{i+1} (ID: {shape_id})...")
        
        # توليد العينة
        sample = generate_test_sample(shape_id)
        
        # التنبؤ
        with torch.no_grad():
            src_img = sample['source_image'].unsqueeze(0).to(DEVICE)
            # src_cam = تم إزالته! 🎯
            tgt_cam = sample['target_camera'].unsqueeze(0).to(DEVICE)
            
            pred_img = model(src_img, tgt_cam)  # 🎯 فقط target_cam!
        
        # حساب المقاييس
        metrics = calculate_metrics(pred_img[0].cpu(), sample['target_image'])
        all_metrics.append(metrics)
        
        print(f"   PSNR: {metrics['psnr']:.2f} dB | Similarity: {metrics['similarity']:.4f}")
        
        # تحضير الصور للعرض
        src_np = sample['source_image'].permute(1, 2, 0).numpy()
        pred_np = pred_img[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        tgt_np = sample['target_image'].permute(1, 2, 0).numpy()
        diff_np = np.abs(pred_np - tgt_np)
        
        # رسم الصور
        axes[i, 0].imshow(src_np)
        axes[i, 0].set_title(f"Source\n(Elev: {sample['src_view'][0]:.1f}°, Azim: {sample['src_view'][1]:.1f}°)", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_np)
        axes[i, 1].set_title(f"Prediction\nPSNR: {metrics['psnr']:.2f} dB", fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(tgt_np)
        axes[i, 2].set_title(f"Ground Truth\n(Elev: {sample['tgt_view'][0]:.1f}°, Azim: {sample['tgt_view'][1]:.1f}°)", fontsize=10)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(diff_np)
        axes[i, 3].set_title(f"Difference\nSim: {metrics['similarity']:.4f}", fontsize=10)
        axes[i, 3].axis('off')
    
    # حساب المتوسطات
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_sim = np.mean([m['similarity'] for m in all_metrics])
    
    plt.suptitle(f"Phase 4 Model Test Results\nAvg PSNR: {avg_psnr:.2f} dB | Avg Similarity: {avg_sim:.4f}", 
                 fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # حفظ النتائج
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(OUTPUT_DIR, f'test_results_{timestamp}.png')
    plt.savefig(result_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"   Average PSNR: {avg_psnr:.2f} dB")
    print(f"   Average Similarity: {avg_sim:.4f}")
    print(f"\n💾 Results saved to: {result_path}")
    print("=" * 60)
    
    return all_metrics

def main():
    print("=" * 60)
    print("🚀 Phase 4 U-Net Model Testing")
    print("=" * 60)
    
    # تحميل النموذج
    model = load_model()
    
    # اختبار النموذج
    metrics = test_model(model, num_samples=NUM_TEST_SAMPLES)
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
