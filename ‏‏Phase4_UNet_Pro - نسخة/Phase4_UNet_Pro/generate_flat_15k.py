"""
Generate Flat 15k Dataset - FINAL VERSION 🚀
يولد 15,000 صورة فريدة بجودة عالية وتنوع كبير
المميزات:
- دقة 128x128
- حواف رفيعة ورمادية خفيفة (كما في المثال)
- تنوع في عدد الأضلاع (4, 5, 6, 7, 8 أضلاع)
- تنوع بسيط في الطول والعرض
- معالجة متوازية (11 كور)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
import random

# ============== إعدادات التوليد ==============
OUTPUT_DIR = "dataset_15k"
NUM_SHAPES = 15000
IMAGE_SIZE = 128  # الدقة المطلوبة
NUM_WORKERS = 11
SHAPE_SCALE = 1.2

# ⭐ إعدادات الحواف (رفيعة ورمادية خفيفة)
EDGE_COLOR = (0.45, 0.45, 0.45, 0.6)  # رمادي خفيف شبه شفاف
EDGE_WIDTH = 0.4  # رفيعة جداً
# =============================================


class ImprovedShapeGenerator:
    """مولد أشكال محسّن مع تنوع في عدد الأضلاع"""

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_box(self, width=1.0, depth=1.0, height=1.0):
        """صندوق (مكعب مع تحكم بالأبعاد الثلاثة)"""
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

    def generate_ngon_pyramid(self, n_sides=4, radius=0.6, height=1.0):
        """هرم بعدد أضلاع متغير (4, 5, 6, 7, 8)"""
        angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
        
        base = np.array([[radius * np.cos(a), radius * np.sin(a), -height/2] for a in angles])
        apex = np.array([0, 0, height/2])
        
        faces = []
        faces.append([base[i] for i in range(n_sides)])
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append([base[i], base[j], apex])
        
        name = f"Pyramid{n_sides}"
        return faces, name, "poly"

    def generate_ngon_prism(self, n_sides=6, radius=0.5, height=1.2):
        """منشور بعدد أضلاع متغير (4, 5, 6, 7, 8)"""
        h = height / 2
        angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1]
        
        bottom = np.array([[radius * np.cos(a), radius * np.sin(a), -h] for a in angles])
        top = np.array([[radius * np.cos(a), radius * np.sin(a), h] for a in angles])
        
        faces = []
        faces.append([bottom[i] for i in range(n_sides)])
        faces.append([top[i] for i in range(n_sides)])
        for i in range(n_sides):
            j = (i + 1) % n_sides
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        
        name = f"Prism{n_sides}"
        return faces, name, "poly"

    def generate_octahedron(self, width=1.0, height=1.0):
        """ثماني أوجه مع تحكم بالنسب"""
        w = width * 0.6
        h = height * 0.6
        vertices = np.array([
            [0, 0, h], [0, 0, -h],
            [w, 0, 0], [-w, 0, 0],
            [0, w, 0], [0, -w, 0]
        ])
        faces = [
            [vertices[0], vertices[2], vertices[4]],
            [vertices[0], vertices[4], vertices[3]],
            [vertices[0], vertices[3], vertices[5]],
            [vertices[0], vertices[5], vertices[2]],
            [vertices[1], vertices[4], vertices[2]],
            [vertices[1], vertices[3], vertices[4]],
            [vertices[1], vertices[5], vertices[3]],
            [vertices[1], vertices[2], vertices[5]],
        ]
        return faces, "Octahedron", "poly"

    def generate_sphere(self, radius=0.6, resolution=16):
        """كرة"""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return (x, y, z), "Sphere", "surface"

    def generate_cylinder(self, radius=0.5, height=1.5, resolution=16):
        """أسطوانة"""
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.array([-height/2, height/2])
        Theta, Z = np.meshgrid(theta, z)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)
        return (X, Y, Z), "Cylinder", "surface"

    def generate_cone(self, radius=0.6, height=1.5, resolution=16):
        """مخروط"""
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(-height/2, height/2, resolution)
        Theta, Z = np.meshgrid(theta, z)
        R = radius * (1 - (Z + height/2) / height)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        return (X, Y, Z), "Cone", "surface"

    def generate_torus(self, major_radius=0.6, minor_radius=0.25, resolution=16):
        """حلقة"""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, 2 * np.pi, resolution)
        U, V = np.meshgrid(u, v)
        X = (major_radius + minor_radius * np.cos(V)) * np.cos(U)
        Y = (major_radius + minor_radius * np.cos(V)) * np.sin(U)
        Z = minor_radius * np.sin(V)
        return (X, Y, Z), "Torus", "surface"

    def generate_random_shape(self):
        """توليد شكل عشوائي مع تنوع في الأبعاد وعدد الأضلاع"""
        
        shape_category = random.choice([
            'box', 'pyramid', 'prism', 'octahedron',
            'sphere', 'cylinder', 'cone', 'torus'
        ])
        
        if shape_category == 'box':
            w = np.random.uniform(0.7, 1.1)
            d = np.random.uniform(0.7, 1.1)
            h = np.random.uniform(0.6, 1.3)
            return self.generate_box(width=w, depth=d, height=h)
            
        elif shape_category == 'pyramid':
            n_sides = random.choice([4, 5, 6, 7, 8])
            radius = np.random.uniform(0.5, 0.8)
            height = np.random.uniform(0.7, 1.4)
            return self.generate_ngon_pyramid(n_sides=n_sides, radius=radius, height=height)
            
        elif shape_category == 'prism':
            n_sides = random.choice([4, 5, 6, 7, 8])
            radius = np.random.uniform(0.4, 0.7)
            height = np.random.uniform(0.6, 1.6)
            return self.generate_ngon_prism(n_sides=n_sides, radius=radius, height=height)
            
        elif shape_category == 'octahedron':
            width = np.random.uniform(0.8, 1.2)
            height = np.random.uniform(0.8, 1.3)
            return self.generate_octahedron(width=width, height=height)
            
        elif shape_category == 'sphere':
            radius = np.random.uniform(0.45, 0.65)
            return self.generate_sphere(radius=radius)
            
        elif shape_category == 'cylinder':
            radius = np.random.uniform(0.3, 0.55)
            height = np.random.uniform(0.7, 1.6)
            return self.generate_cylinder(radius=radius, height=height)
            
        elif shape_category == 'cone':
            radius = np.random.uniform(0.4, 0.65)
            height = np.random.uniform(0.8, 1.5)
            return self.generate_cone(radius=radius, height=height)
            
        else:  # torus
            major = np.random.uniform(0.45, 0.6)
            minor = np.random.uniform(0.12, 0.28)
            return self.generate_torus(major_radius=major, minor_radius=minor)


def render_single_shape(args):
    """دالة التوليد لكل صورة"""
    idx, output_dir = args
    
    np.random.seed(idx + 54321)
    random.seed(idx + 54321)
    
    generator = ImprovedShapeGenerator()
    shape_data, shape_type, render_type = generator.generate_random_shape()
    
    elev = np.random.uniform(15, 55)
    azim = np.random.uniform(0, 360)
    
    # لون رمادي للجسم
    gray = np.random.uniform(0.5, 0.7)
    color = (gray, gray, gray)
    
    # إعداد الشكل - حجم ثابت 128×128 بالضبط
    # figsize=1.28 inch × dpi=100 = 128 pixels
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    fig.patch.set_facecolor('white')
    
    # ملء الشكل بالكامل (بدون هوامش)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_facecolor('white')
    
    if render_type == "poly":
        poly = Poly3DCollection(
            shape_data,
            facecolors=color,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            alpha=1.0
        )
        ax.add_collection3d(poly)
    else:
        x, y, z = shape_data
        ax.plot_surface(
            x, y, z,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=1.0,
            shade=False
        )
    
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
    
    # حفظ مباشر بحجم 128×128
    filename = f"shape_{idx:05d}_{shape_type}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=100, facecolor='white')
    plt.close(fig)
    
    return idx


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("=" * 70)
    print("🎨 توليد Dataset ثلاثي الأبعاد - النسخة النهائية")
    print("=" * 70)
    print(f"📊 الإعدادات:")
    print(f"   - عدد الأشكال: {NUM_SHAPES:,}")
    print(f"   - الدقة: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   - عدد الكور: {NUM_WORKERS}")
    print(f"   - تنوع الأضلاع: 4, 5, 6, 7, 8 أضلاع")
    print(f"   - الحواف: رفيعة ورمادية خفيفة")
    print(f"   - المجلد: {OUTPUT_DIR}/")
    print("=" * 70)
    
    start_time = time.time()
    
    tasks = [(i, OUTPUT_DIR) for i in range(NUM_SHAPES)]
    
    print(f"\n🚀 بدء التوليد باستخدام {NUM_WORKERS} كور...")
    
    completed = 0
    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(render_single_shape, tasks, chunksize=50):
            completed += 1
            if completed % 500 == 0 or completed == NUM_SHAPES:
                elapsed = time.time() - start_time
                speed = completed / elapsed
                remaining = (NUM_SHAPES - completed) / speed if speed > 0 else 0
                print(f"   📸 [{completed:,}/{NUM_SHAPES:,}] "
                      f"({completed/NUM_SHAPES*100:.1f}%) | "
                      f"⚡ {speed:.1f} img/s | "
                      f"⏳ {remaining/60:.1f} min remaining")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ اكتمل التوليد بنجاح!")
    print(f"⏱️  الوقت الإجمالي: {total_time/60:.1f} دقيقة")
    print(f"⚡ متوسط السرعة: {NUM_SHAPES/total_time:.1f} صورة/ثانية")
    print(f"📁 الصور في: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
