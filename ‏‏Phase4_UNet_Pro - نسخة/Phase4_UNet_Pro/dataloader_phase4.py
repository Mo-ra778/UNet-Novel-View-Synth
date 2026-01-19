"""
Phase 4 DataLoader - Full Coverage Edition 🧠
نظام توليد البيانات اللحظي (On-the-Fly) - النسخة المحسنة
- 15,000 شكل ثابت (نفس الشكل = نفس الهندسة دائماً)
- 60 زاوية لكل شكل (Stratified Random Distribution)
- تغطية كاملة للنطاق: Elevation 0°-90° | Azimuth 0°-360°
- التدريب على أزواج من الزوايا المختلفة
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import BytesIO
from PIL import Image
import random

# ============== الإعدادات ==============
NUM_SHAPES = 15000            # عدد الأشكال الثابتة
IMAGE_SIZE = 128              # أبعاد الصورة
VIEWS_PER_SHAPE = 60          # 60 زاوية لكل شكل
NUM_STRATA = 6                # عدد الطبقات (0-90 مقسمة على 6)
VIEWS_PER_STRATUM = 10        # عينات لكل طبقة (60÷6=10)
EDGE_COLOR = (0.45, 0.45, 0.45, 0.6)
EDGE_WIDTH = 0.4
SHAPE_SCALE = 1.2
# ======================================


def generate_stratified_random_views(num_views=60, seed=None):
    """
    توليد زوايا عشوائية موزعة بشكل منظم 🎯
    
    الاستراتيجية:
    - تقسيم Elevation (0°-90°) إلى 6 طبقات متساوية
    - في كل طبقة: توليد 10 زوايا عشوائية
    - Azimuth: عشوائي كامل 0°-360° لكل عينة
    
    المزايا:
    ✅ تغطية شاملة لكامل النطاق
    ✅ عشوائية تمنع overfitting
    ✅ توازن مثالي - لا توجد مناطق مهملة
    
    Args:
        num_views: إجمالي عدد الزوايا (افتراضي: 60)
        seed: للتكرارية (None = عشوائي كامل)
    
    Returns:
        list: قائمة من (elevation, azimuth) tuples
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    views = []
    
    # تقسيم النطاق 0-90 إلى طبقات متساوية
    strata_ranges = []
    for i in range(NUM_STRATA):
        min_elev = i * (90.0 / NUM_STRATA)
        max_elev = (i + 1) * (90.0 / NUM_STRATA)
        strata_ranges.append((min_elev, max_elev))
    
    # توليد عينات عشوائية في كل طبقة
    for min_elev, max_elev in strata_ranges:
        for _ in range(VIEWS_PER_STRATUM):
            elevation = np.random.uniform(min_elev, max_elev)
            azimuth = np.random.uniform(0, 360)
            views.append((elevation, azimuth))
    
    # خلط الزوايا لتنوع أكبر
    random.shuffle(views)
    
    return views


def generate_fibonacci_views(num_views=60):
    """
    [DEPRECATED - للتوافق مع الكود القديم فقط]
    توليد 60 زاوية موزعة بالتساوي على الكرة (Fibonacci)
    
    ⚠️ تحذير: هذه الدالة قديمة ومحدودة (10°-70° فقط)
    استخدم generate_stratified_random_views() بدلاً منها
    """
    views = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    
    for i in range(num_views):
        y = 1 - (i / float(num_views - 1)) * 2
        theta = phi * i
        elevation = np.degrees(np.arcsin(y))
        azimuth = np.degrees(theta) % 360
        elevation = 10 + (elevation + 90) / 180 * 60  # محدود: 10°-70°
        views.append((elevation, azimuth))
    
    return views


class FixedShapeGenerator:
    """
    مولد أشكال ثابت - نفس الـ shape_id = نفس الشكل دائماً
    """
    
    def __init__(self):
        pass
    
    def generate_shape_by_id(self, shape_id):
        """
        توليد شكل بناءً على ID ثابت
        نفس الـ ID = نفس الشكل بالضبط
        """
        # تثبيت العشوائية بناءً على shape_id
        rng = np.random.RandomState(shape_id + 12345)
        py_rng = random.Random(shape_id + 12345)
        
        # اختيار نوع الشكل
        shape_types = ['box', 'pyramid', 'prism', 'octahedron', 
                       'sphere', 'cylinder', 'cone', 'torus']
        shape_type = py_rng.choice(shape_types)
        
        # توليد معاملات ثابتة لهذا الشكل
        if shape_type == 'box':
            return self._generate_box(
                width=rng.uniform(0.7, 1.1),
                depth=rng.uniform(0.7, 1.1),
                height=rng.uniform(0.6, 1.3)
            )
        elif shape_type == 'pyramid':
            return self._generate_ngon_pyramid(
                n_sides=py_rng.choice([4, 5, 6, 7, 8]),
                radius=rng.uniform(0.5, 0.8),
                height=rng.uniform(0.7, 1.4)
            )
        elif shape_type == 'prism':
            return self._generate_ngon_prism(
                n_sides=py_rng.choice([4, 5, 6, 7, 8]),
                radius=rng.uniform(0.4, 0.7),
                height=rng.uniform(0.6, 1.6)
            )
        elif shape_type == 'octahedron':
            return self._generate_octahedron(
                width=rng.uniform(0.8, 1.2),
                height=rng.uniform(0.8, 1.3)
            )
        elif shape_type == 'sphere':
            return self._generate_sphere(radius=rng.uniform(0.45, 0.65))
        elif shape_type == 'cylinder':
            return self._generate_cylinder(
                radius=rng.uniform(0.3, 0.55),
                height=rng.uniform(0.7, 1.6)
            )
        elif shape_type == 'cone':
            return self._generate_cone(
                radius=rng.uniform(0.4, 0.65),
                height=rng.uniform(0.8, 1.5)
            )
        else:
            return self._generate_torus(
                major_radius=rng.uniform(0.45, 0.6),
                minor_radius=rng.uniform(0.12, 0.28)
            )
    
    def get_shape_color(self, shape_id):
        """لون ثابت لكل شكل"""
        rng = np.random.RandomState(shape_id + 99999)
        gray = rng.uniform(0.5, 0.7)
        return (gray, gray, gray)

    # ========== دوال توليد الأشكال ==========
    
    def _generate_box(self, width=1.0, depth=1.0, height=1.0):
        w, d, h = width/2, depth/2, height/2
        vertices = np.array([
            [-w, -d, -h], [w, -d, -h], [w, d, -h], [-w, d, -h],
            [-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
        ]
        return faces, "Box", "poly"

    def _generate_ngon_pyramid(self, n_sides=4, radius=0.6, height=1.0):
        angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
        base = np.array([[radius * np.cos(a), radius * np.sin(a), -height/2] for a in angles])
        apex = np.array([0, 0, height/2])
        faces = [[base[i] for i in range(n_sides)]]
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append([base[i], base[j], apex])
        return faces, f"Pyramid{n_sides}", "poly"

    def _generate_ngon_prism(self, n_sides=6, radius=0.5, height=1.2):
        h = height / 2
        angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
        bottom = np.array([[radius * np.cos(a), radius * np.sin(a), -h] for a in angles])
        top = np.array([[radius * np.cos(a), radius * np.sin(a), h] for a in angles])
        faces = [[bottom[i] for i in range(n_sides)], [top[i] for i in range(n_sides)]]
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        return faces, f"Prism{n_sides}", "poly"

    def _generate_octahedron(self, width=1.0, height=1.0):
        w, h = width * 0.6, height * 0.6
        vertices = np.array([
            [0, 0, h], [0, 0, -h], [w, 0, 0], [-w, 0, 0], [0, w, 0], [0, -w, 0]
        ])
        faces = [
            [vertices[0], vertices[2], vertices[4]], [vertices[0], vertices[4], vertices[3]],
            [vertices[0], vertices[3], vertices[5]], [vertices[0], vertices[5], vertices[2]],
            [vertices[1], vertices[4], vertices[2]], [vertices[1], vertices[3], vertices[4]],
            [vertices[1], vertices[5], vertices[3]], [vertices[1], vertices[2], vertices[5]],
        ]
        return faces, "Octahedron", "poly"

    def _generate_sphere(self, radius=0.6, resolution=16):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return (x, y, z), "Sphere", "surface"

    def _generate_cylinder(self, radius=0.5, height=1.5, resolution=16):
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.array([-height/2, height/2])
        Theta, Z = np.meshgrid(theta, z)
        X, Y = radius * np.cos(Theta), radius * np.sin(Theta)
        return (X, Y, Z), "Cylinder", "surface"

    def _generate_cone(self, radius=0.6, height=1.5, resolution=16):
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(-height/2, height/2, resolution)
        Theta, Z = np.meshgrid(theta, z)
        R = radius * (1 - (Z + height/2) / height)
        X, Y = R * np.cos(Theta), R * np.sin(Theta)
        return (X, Y, Z), "Cone", "surface"

    def _generate_torus(self, major_radius=0.6, minor_radius=0.25, resolution=16):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, 2 * np.pi, resolution)
        U, V = np.meshgrid(u, v)
        X = (major_radius + minor_radius * np.cos(V)) * np.cos(U)
        Y = (major_radius + minor_radius * np.cos(V)) * np.sin(U)
        Z = minor_radius * np.sin(V)
        return (X, Y, Z), "Torus", "surface"


def render_shape(shape_data, elev, azim, color):
    """رسم الشكل وإرجاع Tensor"""
    faces, shape_type, render_type = shape_data
    
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_facecolor('white')
    
    if render_type == "poly":
        poly = Poly3DCollection(faces, facecolors=color, edgecolors=EDGE_COLOR,
                                linewidths=EDGE_WIDTH, alpha=1.0)
        ax.add_collection3d(poly)
    else:
        x, y, z = faces
        ax.plot_surface(x, y, z, color=color, edgecolor=EDGE_COLOR,
                       linewidth=EDGE_WIDTH, alpha=1.0, shade=False)
    
    limit = 1.0 / SHAPE_SCALE
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf)
    img = img.convert('RGB')
    tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    buf.close()
    
    return tensor


class Phase4Dataset(Dataset):
    """
    Dataset ثابت الأشكال مع تغطية كاملة للزوايا ⭐
    
    - 15,000 شكل ثابت (نفس الـ ID = نفس الشكل)
    - 60 زاوية لكل شكل (عشوائية موزعة بشكل منظم)
    - تغطية كاملة: Elevation 0°-90° | Azimuth 0°-360°
    - التدريب على أزواج: (زاوية_مصدر, زاوية_هدف)
    
    الجديد في هذه النسخة:
    ✅ زوايا جديدة في كل epoch (عبر epoch_seed)
    ✅ تغطية شاملة لكامل النطاق (0°-90°)
    ✅ توزيع منظم (Stratified Random)
    
    Total Pairs = 15,000 × pairs_per_shape
    """
    
    def __init__(self, num_shapes=NUM_SHAPES, pairs_per_shape=10, epoch_seed=None):
        """
        num_shapes: عدد الأشكال الثابتة (15,000)
        pairs_per_shape: عدد الأزواج لكل شكل في كل epoch
        epoch_seed: seed للزوايا (يتغير كل epoch)
                   None = عشوائي كامل
                   number = قابل للتكرار
        """
        self.num_shapes = num_shapes
        self.pairs_per_shape = pairs_per_shape
        self.epoch_seed = epoch_seed
        
        # توليد الزوايا باستخدام التوزيع الجديد 🎯
        self.views = generate_stratified_random_views(
            num_views=VIEWS_PER_SHAPE, 
            seed=epoch_seed
        )
        
        self.generator = FixedShapeGenerator()
        
        # حجم Dataset = عدد الأشكال × عدد الأزواج لكل شكل
        self.total_samples = num_shapes * pairs_per_shape
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # تحديد الشكل والزوج
        shape_id = idx // self.pairs_per_shape
        pair_id = idx % self.pairs_per_shape
        
        # توليد الشكل (ثابت بناءً على shape_id)
        shape_data = self.generator.generate_shape_by_id(shape_id)
        shape_color = self.generator.get_shape_color(shape_id)
        
        # اختيار زاويتين عشوائيتين (مختلفتين)
        # نستخدم pair_id لجعل الاختيار متكرر
        pair_rng = random.Random(shape_id * 1000 + pair_id + idx)
        src_idx, tgt_idx = pair_rng.sample(range(VIEWS_PER_SHAPE), 2)
        
        src_elev, src_azim = self.views[src_idx]
        tgt_elev, tgt_azim = self.views[tgt_idx]
        
        # رسم الشكل من الزاويتين (بنفس اللون!)
        source_tensor = render_shape(shape_data, src_elev, src_azim, shape_color)
        target_tensor = render_shape(shape_data, tgt_elev, tgt_azim, shape_color)
        
        # معلومات كاميرا الهدف فقط (تطبيع)
        # 🎯 التحسين: لا نحتاج source_camera!
        target_cam = torch.tensor([
            tgt_elev / 90.0,
            tgt_azim / 180.0 - 1.0,
            1.0
        ], dtype=torch.float32)
        
        return {
            'source_image': source_tensor,
            # 'source_camera': تم إزالته! النموذج يستنتجه من الصورة 🧠
            'target_image': target_tensor,
            'target_camera': target_cam,
            'shape_id': shape_id
        }


def create_phase4_dataloaders(batch_size=16, num_workers=4, pairs_per_shape=10, epoch_seed=None):
    """
    إنشاء DataLoaders للتدريب مع التغطية الكاملة ⭐
    
    Args:
        batch_size: حجم الـ batch
        num_workers: عدد عمليات التوليد المتوازية
        pairs_per_shape: عدد أزواج الزوايا لكل شكل
        epoch_seed: seed للزوايا (مختلف لكل epoch)
                   استخدم: base_seed + epoch_number
    
    Returns:
        train_loader, val_loader
    
    ملاحظات:
    - الزوايا تتغير في كل epoch (عبر epoch_seed)
    - تغطية كاملة: Elevation 0°-90° | Azimuth 0°-360°
    - 60 زاوية موزعة على 6 طبقات (10 لكل طبقة)
    """
    
    # Training: 15,000 شكل × 10 أزواج = 150,000 عينة
    train_dataset = Phase4Dataset(
        num_shapes=NUM_SHAPES,
        pairs_per_shape=pairs_per_shape,
        epoch_seed=epoch_seed  # زوايا جديدة كل epoch!
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Validation: 500 شكل × 5 أزواج = 2,500 عينة
    # نستخدم seed ثابت للـ validation للمقارنة
    val_dataset = Phase4Dataset(
        num_shapes=500,
        pairs_per_shape=5,
        epoch_seed=42  # ثابت للتحقق
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Unique Shapes: {NUM_SHAPES:,}")
    print(f"   - Views per Shape: {VIEWS_PER_SHAPE}")
    print(f"   - Training Samples: {len(train_dataset):,}")
    print(f"   - Validation Samples: {len(val_dataset):,}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("🧪 Testing Fixed 15K DataLoader...")
    
    loader, _ = create_phase4_dataloaders(batch_size=4, num_workers=0, pairs_per_shape=2)
    
    print(f"\nDataset size: {len(loader.dataset)}")
    
    for i, batch in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"  Source Image: {batch['source_image'].shape}")
        print(f"  Target Image: {batch['target_image'].shape}")
        print(f"  Shape IDs: {batch['shape_id'].tolist()}")
        
        if i >= 2:
            break
    
    # اختبار أن نفس shape_id يعطي نفس الشكل
    print("\n🔍 Testing shape consistency...")
    gen = FixedShapeGenerator()
    shape1 = gen.generate_shape_by_id(100)
    shape2 = gen.generate_shape_by_id(100)
    print(f"  Same shape for ID=100? {shape1[1] == shape2[1]}")  # نفس النوع
    
    print("\n✅ DataLoader working correctly!")
