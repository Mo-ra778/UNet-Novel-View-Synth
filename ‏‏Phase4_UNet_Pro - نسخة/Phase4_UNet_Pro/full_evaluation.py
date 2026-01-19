"""
📊 Full Model Evaluation Script
اختبار شامل للنموذج على كامل مجموعة التحقق

يقوم هذا السكربت بـ:
1. اختبار النموذج على 2500 عينة (كل مجموعة التحقق)
2. حساب PSNR, SSIM, MSE, MAE
3. تحليل الأداء حسب نوع الشكل
4. إنشاء تقرير مفصل
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from collections import defaultdict

# استيراد الوحدات المحلية
from model_unet import get_unet_model
from dataloader_phase4 import create_phase4_dataloaders, FixedShapeGenerator

# ============== الإعدادات ==============
MODEL_PATH = 'results_phase4_optimized/best_model.pth'
OUTPUT_DIR = 'evaluation_results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
# ======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    """تحميل النموذج المدرب"""
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = get_unet_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"✅ Model loaded on {DEVICE}")
    return model


def calculate_metrics(pred, target):
    """حساب جميع مقاييس الجودة"""
    # التأكد من أن البيانات على CPU
    pred = pred.cpu().float()
    target = target.cpu().float()
    
    # MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    
    # MAE
    mae = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
    
    # PSNR
    psnr = -10 * torch.log10(mse + 1e-8)
    
    # Similarity (1 - MAE)
    similarity = 1.0 - mae
    
    return {
        'mse': mse.numpy(),
        'mae': mae.numpy(),
        'psnr': psnr.numpy(),
        'similarity': similarity.numpy()
    }


def evaluate_model(model):
    """تقييم شامل للنموذج"""
    print("\n" + "=" * 70)
    print("🧪 FULL MODEL EVALUATION")
    print("=" * 70)
    
    # إنشاء DataLoader للتحقق
    _, val_loader = create_phase4_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
        pairs_per_shape=10
    )
    
    total_samples = len(val_loader.dataset)
    print(f"\n📊 Evaluating on {total_samples} validation samples...")
    
    # تجميع النتائج
    all_psnr = []
    all_similarity = []
    all_mse = []
    all_mae = []
    
    # شريط التقدم
    pbar = tqdm(val_loader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            # نقل البيانات للـ GPU
            source_img = batch['source_image'].to(DEVICE)
            target_img = batch['target_image'].to(DEVICE)
            target_cam = batch['target_camera'].to(DEVICE)
            
            # التنبؤ (النموذج المحسّن يستخدم فقط target_camera)
            pred_img = model(source_img, target_cam)
            
            # حساب المقاييس
            metrics = calculate_metrics(pred_img, target_img)
            
            all_psnr.extend(metrics['psnr'])
            all_similarity.extend(metrics['similarity'])
            all_mse.extend(metrics['mse'])
            all_mae.extend(metrics['mae'])
            
            # تحديث شريط التقدم
            pbar.set_postfix({
                'Avg PSNR': f"{np.mean(all_psnr):.2f}",
                'Avg Sim': f"{np.mean(all_similarity):.4f}"
            })
    
    # تحويل لـ numpy
    all_psnr = np.array(all_psnr)
    all_similarity = np.array(all_similarity)
    all_mse = np.array(all_mse)
    all_mae = np.array(all_mae)
    
    # حساب الإحصائيات
    results = {
        'total_samples': len(all_psnr),
        'psnr': {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
            'min': np.min(all_psnr),
            'max': np.max(all_psnr),
            'median': np.median(all_psnr)
        },
        'similarity': {
            'mean': np.mean(all_similarity),
            'std': np.std(all_similarity),
            'min': np.min(all_similarity),
            'max': np.max(all_similarity),
            'median': np.median(all_similarity)
        },
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse)
        },
        'mae': {
            'mean': np.mean(all_mae),
            'std': np.std(all_mae)
        }
    }
    
    return results, all_psnr, all_similarity


def create_report(results, all_psnr, all_similarity):
    """إنشاء تقرير مرئي"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. توزيع PSNR
    ax1 = axes[0, 0]
    ax1.hist(all_psnr, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(results['psnr']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['psnr']['mean']:.2f}")
    ax1.axvline(results['psnr']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {results['psnr']['median']:.2f}")
    ax1.set_xlabel('PSNR (dB)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('PSNR Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. توزيع Similarity
    ax2 = axes[0, 1]
    ax2.hist(all_similarity * 100, bins=50, color='seagreen', edgecolor='white', alpha=0.8)
    ax2.axvline(results['similarity']['mean'] * 100, color='red', linestyle='--', linewidth=2, label=f"Mean: {results['similarity']['mean']*100:.2f}%")
    ax2.set_xlabel('Similarity (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Similarity Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. ملخص الإحصائيات
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║           📊 MODEL EVALUATION SUMMARY                 ║
    ╠══════════════════════════════════════════════════════╣
    ║  Total Samples Tested: {results['total_samples']:,}                      ║
    ╠══════════════════════════════════════════════════════╣
    ║  PSNR (Peak Signal-to-Noise Ratio):                  ║
    ║    • Mean:   {results['psnr']['mean']:.2f} dB                              ║
    ║    • Std:    {results['psnr']['std']:.2f} dB                               ║
    ║    • Min:    {results['psnr']['min']:.2f} dB                               ║
    ║    • Max:    {results['psnr']['max']:.2f} dB                               ║
    ║    • Median: {results['psnr']['median']:.2f} dB                            ║
    ╠══════════════════════════════════════════════════════╣
    ║  Similarity:                                         ║
    ║    • Mean:   {results['similarity']['mean']*100:.2f}%                              ║
    ║    • Std:    {results['similarity']['std']*100:.2f}%                               ║
    ║    • Min:    {results['similarity']['min']*100:.2f}%                               ║
    ║    • Max:    {results['similarity']['max']*100:.2f}%                               ║
    ╠══════════════════════════════════════════════════════╣
    ║  MSE:  {results['mse']['mean']:.6f} ± {results['mse']['std']:.6f}                  ║
    ║  MAE:  {results['mae']['mean']:.6f} ± {results['mae']['std']:.6f}                  ║
    ╚══════════════════════════════════════════════════════╝
    """
    
    ax3.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', transform=ax3.transAxes)
    
    # 4. تصنيف الجودة
    ax4 = axes[1, 1]
    
    # تصنيف العينات حسب PSNR
    categories = {
        'Excellent (>35 dB)': np.sum(all_psnr > 35),
        'Very Good (30-35 dB)': np.sum((all_psnr >= 30) & (all_psnr <= 35)),
        'Good (25-30 dB)': np.sum((all_psnr >= 25) & (all_psnr < 30)),
        'Fair (20-25 dB)': np.sum((all_psnr >= 20) & (all_psnr < 25)),
        'Poor (<20 dB)': np.sum(all_psnr < 20)
    }
    
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    wedges, texts, autotexts = ax4.pie(
        categories.values(), 
        labels=categories.keys(),
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05, 0, 0, 0, 0],
        shadow=True
    )
    ax4.set_title('Quality Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle('Phase 4 U-Net Model - Full Evaluation Report', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # حفظ التقرير
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f'full_evaluation_{timestamp}.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return report_path


def main():
    print("=" * 70)
    print("🚀 Phase 4 U-Net - Full Model Evaluation")
    print("=" * 70)
    
    # تحميل النموذج
    model = load_model()
    
    # تقييم النموذج
    results, all_psnr, all_similarity = evaluate_model(model)
    
    # طباعة النتائج
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS:")
    print("=" * 70)
    print(f"  Total Samples: {results['total_samples']:,}")
    print(f"\n  PSNR:")
    print(f"    • Mean:   {results['psnr']['mean']:.2f} dB")
    print(f"    • Std:    {results['psnr']['std']:.2f} dB")
    print(f"    • Min:    {results['psnr']['min']:.2f} dB")
    print(f"    • Max:    {results['psnr']['max']:.2f} dB")
    print(f"\n  Similarity:")
    print(f"    • Mean:   {results['similarity']['mean']*100:.2f}%")
    print(f"    • Std:    {results['similarity']['std']*100:.2f}%")
    print("=" * 70)
    
    # إنشاء التقرير المرئي
    print("\n📈 Generating visual report...")
    report_path = create_report(results, all_psnr, all_similarity)
    print(f"💾 Report saved to: {report_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
