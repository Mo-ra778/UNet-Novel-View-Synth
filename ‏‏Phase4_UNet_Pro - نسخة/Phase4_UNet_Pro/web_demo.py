"""
🎮 Phase 4 U-Net Web Demo
واجهة تفاعلية لاختبار النموذج
"""

import torch
import numpy as np
import gradio as gr
from PIL import Image
import os

# استيراد الوحدات المحلية
from model_unet import get_unet_model
from dataloader_phase4 import FixedShapeGenerator, render_shape, generate_stratified_random_views

# ============== الإعدادات ==============
MODEL_PATH = 'results_phase4_optimized/best_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ======================================

# تحميل النموذج
print("📦 Loading model...")
model = get_unet_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# مولد الأشكال
generator = FixedShapeGenerator()


def generate_random_shape(shape_id):
    """توليد شكل عشوائي"""
    shape_data = generator.generate_shape_by_id(shape_id)
    shape_color = generator.get_shape_color(shape_id)
    
    # رسم من زاوية عشوائية
    views = generate_stratified_random_views(60, seed=shape_id)
    view = views[0]
    
    img_tensor = render_shape(shape_data, view[0], view[1], shape_color)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np), view[0], view[1]


def predict_view(source_image, target_elevation, target_azimuth):
    """التنبؤ بالصورة من الزاوية المطلوبة"""
    if source_image is None:
        return None, "⚠️ الرجاء رفع صورة أو توليد شكل عشوائي"
    
    # تحويل الصورة لـ tensor
    img_np = np.array(source_image).astype(np.float32) / 255.0
    
    # التأكد من أن الصورة RGB
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    
    # تغيير الحجم إلى 128x128
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((img_np * 255).astype(np.uint8))
    img_pil = img_pil.resize((128, 128), PILImage.LANCZOS)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    
    # تحويل لـ tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # تحضير كاميرا الهدف
    target_cam = torch.tensor([
        target_elevation / 90.0,
        target_azimuth / 180.0 - 1.0,
        1.0
    ], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # التنبؤ
    with torch.no_grad():
        pred = model(img_tensor, target_cam)
    
    # تحويل النتيجة لصورة
    pred_np = pred[0].cpu().permute(1, 2, 0).numpy()
    pred_np = np.clip(pred_np, 0, 1)
    pred_np = (pred_np * 255).astype(np.uint8)
    
    info = f"✅ تم التوليد!\n📐 Elevation: {target_elevation:.1f}°\n🔄 Azimuth: {target_azimuth:.1f}°"
    
    return Image.fromarray(pred_np), info


def generate_new_shape():
    """توليد شكل عشوائي جديد"""
    shape_id = np.random.randint(0, 15000)
    img, elev, azim = generate_random_shape(shape_id)
    info = f"🎲 Shape ID: {shape_id}\n📐 Elevation: {elev:.1f}°\n🔄 Azimuth: {azim:.1f}°"
    return img, info


def demo_360_rotation(source_image):
    """توليد دوران 360 درجة"""
    if source_image is None:
        return None, "⚠️ الرجاء رفع صورة أولاً"
    
    images = []
    for azim in range(0, 360, 30):
        pred_img, _ = predict_view(source_image, 30, azim)
        if pred_img:
            images.append(pred_img)
    
    # إنشاء صورة مجمعة
    if images:
        width = 128 * 4
        height = 128 * 3
        combined = Image.new('RGB', (width, height), 'white')
        
        for i, img in enumerate(images):
            x = (i % 4) * 128
            y = (i // 4) * 128
            combined.paste(img.resize((128, 128)), (x, y))
        
        return combined, f"✅ تم توليد 12 زاوية (كل 30°)"
    
    return None, "❌ فشل التوليد"


# إنشاء الواجهة
with gr.Blocks(title="Phase 4 U-Net Demo") as demo:
    gr.Markdown("""
    # 🎮 Phase 4 U-Net - Novel View Synthesis Demo
    ### توليد مناظر جديدة للأشكال ثلاثية الأبعاد
    
    **النتائج:** PSNR = 35.59 dB | Similarity = 99.62%
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 الصورة المصدر")
            source_img = gr.Image(type="pil", label="صورة المصدر")
            
            with gr.Row():
                generate_btn = gr.Button("🎲 توليد شكل عشوائي", variant="secondary")
            
            source_info = gr.Textbox(label="معلومات المصدر", lines=3)
            
        with gr.Column():
            gr.Markdown("### 🎯 إعدادات الزاوية المطلوبة")
            
            elevation = gr.Slider(0, 90, value=45, step=1, label="📐 Elevation (درجة)")
            azimuth = gr.Slider(0, 360, value=180, step=1, label="🔄 Azimuth (درجة)")
            
            predict_btn = gr.Button("🚀 توليد المنظر الجديد", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🖼️ النتيجة")
            output_img = gr.Image(type="pil", label="الصورة المولدة")
            output_info = gr.Textbox(label="معلومات التوليد", lines=3)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            rotation_btn = gr.Button("🔄 توليد دوران 360°", variant="secondary")
        with gr.Column():
            rotation_output = gr.Image(type="pil", label="دوران 360°")
            rotation_info = gr.Textbox(label="معلومات الدوران")
    
    # الأحداث
    generate_btn.click(
        fn=generate_new_shape,
        outputs=[source_img, source_info]
    )
    
    predict_btn.click(
        fn=predict_view,
        inputs=[source_img, elevation, azimuth],
        outputs=[output_img, output_info]
    )
    
    rotation_btn.click(
        fn=demo_360_rotation,
        inputs=[source_img],
        outputs=[rotation_output, rotation_info]
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌐 Starting Web Demo...")
    print("📍 Open: http://localhost:7860")
    print("=" * 50 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
