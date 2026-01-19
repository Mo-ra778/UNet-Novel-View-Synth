"""
🧪 Quick Test - Stratified Random Distribution
اختبار سريع للتأكد من أن التوزيع الجديد يعمل بشكل صحيح
"""

import numpy as np
import matplotlib.pyplot as plt
from dataloader_phase4 import generate_stratified_random_views

print("=" * 70)
print("🧪 Testing Stratified Random Distribution")
print("=" * 70)

# توليد 60 زاوية
views = generate_stratified_random_views(num_views=60, seed=42)

# فصل elevations و azimuths
elevations = [v[0] for v in views]
azimuths = [v[1] for v in views]

# الإحصائيات
print(f"\n📊 Statistics:")
print(f"   Total Views: {len(views)}")
print(f"   Elevation Range: {min(elevations):.1f}° - {max(elevations):.1f}°")
print(f"   Azimuth Range: {min(azimuths):.1f}° - {max(azimuths):.1f}°")

# تحليل التوزيع حسب الطبقات
strata = [
    (0, 15, "0-15°"),
    (15, 30, "15-30°"),
    (30, 45, "30-45°"),
    (45, 60, "45-60°"),
    (60, 75, "60-75°"),
    (75, 90, "75-90°")
]

print(f"\n📐 Distribution by Strata:")
for min_e, max_e, label in strata:
    count = sum(1 for e in elevations if min_e <= e < max_e)
    print(f"   {label}: {count} views")

# رسم توزيع الزوايا
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Histogram - Elevation
ax1 = axes[0]
ax1.hist(elevations, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Target: 10 per stratum')
ax1.set_xlabel('Elevation (°)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Elevation Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histogram - Azimuth
ax2 = axes[1]
ax2.hist(azimuths, bins=20, color='seagreen', edgecolor='white', alpha=0.8)
ax2.set_xlabel('Azimuth (°)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Azimuth Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Scatter - كل الزوايا
ax3 = axes[2]

# تلوين حسب الطبقة
colors = []
for e in elevations:
    if e < 15:
        colors.append('#e74c3c')  # أحمر
    elif e < 30:
        colors.append('#e67e22')  # برتقالي
    elif e < 45:
        colors.append('#f39c12')  # أصفر
    elif e < 60:
        colors.append('#2ecc71')  # أخضر
    elif e < 75:
        colors.append('#3498db')  # أزرق
    else:
        colors.append('#9b59b6')  # بنفسجي

ax3.scatter(azimuths, elevations, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Azimuth (°)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Elevation (°)', fontsize=12, fontweight='bold')
ax3.set_title('All 60 Views (colored by stratum)', fontsize=14, fontweight='bold')
ax3.set_xlim(-10, 370)
ax3.set_ylim(-5, 95)
ax3.grid(True, alpha=0.3)

# إضافة خطوط الطبقات
for i in range(1, 6):
    ax3.axhline(y=i*15, color='gray', linestyle=':', alpha=0.5)

plt.suptitle('🎯 Stratified Random Distribution Test', fontsize=16, fontweight='bold')
plt.tight_layout()

# حفظ
output_path = "distribution_test.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n💾 Visualization saved to: {output_path}")

plt.close()

# اختبار seed reproducibility
print(f"\n🔁 Testing Reproducibility:")
views1 = generate_stratified_random_views(60, seed=123)
views2 = generate_stratified_random_views(60, seed=123)
views3 = generate_stratified_random_views(60, seed=456)

match_12 = all(v1 == v2 for v1, v2 in zip(views1, views2))
match_13 = all(v1 == v3 for v1, v3 in zip(views1, views3))

print(f"   Same seed (123, 123): {'✅ Match' if match_12 else '❌ Mismatch'}")
print(f"   Diff seed (123, 456): {'❌ Same (bad!)' if match_13 else '✅ Different (good!)'}")

# اختبار epoch variation
print(f"\n🎲 Testing Epoch Variation:")
epoch_views = []
for epoch in range(1, 4):
    epoch_seed = 1000 + epoch
    views_ep = generate_stratified_random_views(60, seed=epoch_seed)
    epoch_views.append(views_ep)
    avg_elev = np.mean([v[0] for v in views_ep])
    print(f"   Epoch {epoch} (seed={epoch_seed}): avg_elevation={avg_elev:.1f}°")

# التأكد أن كل epoch مختلف
all_different = True
for i in range(len(epoch_views)):
    for j in range(i+1, len(epoch_views)):
        if epoch_views[i] == epoch_views[j]:
            all_different = False
            break

print(f"   All epochs different: {'✅ Yes' if all_different else '❌ No'}")

print(f"\n{'='*70}")
print("✅ Test Complete!")
print(f"{'='*70}")
