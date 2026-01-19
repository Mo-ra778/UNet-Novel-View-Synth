"""
Visualize All Shapes (Preview) 👁️
يقوم هذا السكريبت بتوليد صفحة HTML تفاعلية تعرض عينة من الأشكال التي سنقوم بتدريب الموديل عليها.
الهدف هو التأكد من تنوع الأشكال وجودتها قبل بدء التدريب الطويل.

ملاحظة: توليد 20,000 صورة وحفظها سيأخد مساحة كبيرة ووقت طويل جداً للمعاينة.
بدلاً من ذلك، سنولد "كتالوج" ذكي يعرض:
1. عينة عشوائية ممثلة لـ 100 شكل (لأن 20 ألف مستحيل فحصها بالعين البشرية).
2. أو إذا أصررت، يمكننا توليدها كلها وحفظها في مجلد.

الخيار التالي يولد 100 شكل منوع في صفحة HTML سهلة التصفح.
"""

import os
import matplotlib.pyplot as plt
from generate_dataset_3d import SimpleShapeGenerator, render_shape_from_pose
import numpy as np
from tqdm import tqdm
import base64
from io import BytesIO

def create_preview_gallery(num_shapes_to_preview=100, output_dir="preview_catalog"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generator = SimpleShapeGenerator(seed=None)
    
    html_content = """
    <html>
    <head>
        <title>Dataset Preview Catalog</title>
        <style>
            body { font-family: sans-serif; background: #f0f0f0; margin: 20px; }
            .grid { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }
            .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 200px; text-align: center; }
            img { width: 100%; height: auto; border-radius: 4px; }
            h1 { text-align: center; color: #333; }
            .stats { text-align: center; margin-bottom: 20px; color: #666; }
        </style>
    </head>
    <body>
        <h1>🎨 Dataset Random Preview</h1>
        <div class="stats">Showing random selection of generated shapes</div>
        <div class="grid">
    """
    
    print(f"🚀 Generating preview for {num_shapes_to_preview} shapes...")
    
    for i in tqdm(range(num_shapes_to_preview)):
        # 1. Generate Shape
        shape_data = generator.generate_random_shape()
        faces, color, shape_type = shape_data
        
        # 2. Render from a nice angle
        elev = 30
        azim = 45
        
        fig = render_shape_from_pose(shape_data, elev, azim, save_path=None, show_title=False)
        
        # 3. Save to memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=80)
        plt.close(fig)
        buf.seek(0)
        
        # 4. Encode to Base64 (to embed directly in HTML)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # 5. Add to HTML
        html_content += f"""
            <div class="card">
                <img src="data:image/png;base64,{img_str}" />
                <p><b>#{i+1}</b><br>{shape_type}</p>
            </div>
        """
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n✅ Created preview gallery at: {os.path.abspath(html_path)}")
    print("🌍 Open this file in your browser to see the shapes!")

if __name__ == "__main__":
    # يمكنك تغيير الرقم هنا إذا أردت رؤية المزيد، لكن 100 كافية لأخذ فكرة
    create_preview_gallery(num_shapes_to_preview=100)
