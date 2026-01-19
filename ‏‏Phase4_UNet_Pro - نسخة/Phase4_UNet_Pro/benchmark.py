"""
Benchmark Script - قياس سرعة التوليد والتدريب 🏎️
"""

import torch
import time
import sys

print("=" * 60)
print("🏎️ Performance Benchmark")
print("=" * 60)

# 1. Check GPU
print("\n📊 System Info:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ⚠️ No GPU detected! Training will be slow.")

# 2. Benchmark DataLoader (Generation Speed)
print("\n" + "=" * 60)
print("📸 Testing Data Generation Speed...")
print("=" * 60)

from dataloader_phase4 import create_phase4_dataloaders

loader, _ = create_phase4_dataloaders(batch_size=16, num_workers=0, pairs_per_shape=2)

# Warmup
print("\n   Warmup (1 batch)...")
for batch in loader:
    _ = batch['source_image']
    break

# Benchmark
print("   Benchmarking (20 batches)...")
start = time.time()
count = 0
total_images = 0

for i, batch in enumerate(loader):
    count += 1
    total_images += batch['source_image'].shape[0] * 2  # source + target
    if count >= 20:
        break

gen_time = time.time() - start
gen_speed = total_images / gen_time

print(f"\n   ✅ Generation Results:")
print(f"      - Batches: {count}")
print(f"      - Images: {total_images}")
print(f"      - Time: {gen_time:.2f}s")
print(f"      - Speed: {gen_speed:.1f} images/s")
print(f"      - Steps/s: {count / gen_time:.2f}")

# 3. Benchmark Training (if GPU available)
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("🧠 Testing Training Speed (GPU)...")
    print("=" * 60)
    
    from model_unet import get_unet_model
    from loss_perceptual import CombinedLoss
    
    device = 'cuda'
    model = get_unet_model(device)
    criterion = CombinedLoss(lambda_perceptual=0.2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy data
    dummy_src = torch.randn(16, 3, 128, 128).to(device)
    dummy_tgt = torch.randn(16, 3, 128, 128).to(device)
    dummy_cam1 = torch.randn(16, 3).to(device)
    dummy_cam2 = torch.randn(16, 3).to(device)
    
    # Warmup
    print("\n   Warmup...")
    for _ in range(5):
        pred = model(dummy_src, dummy_cam1, dummy_cam2)
        loss = criterion(pred, dummy_tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark without FP16
    print("   Testing WITHOUT Mixed Precision...")
    start = time.time()
    for _ in range(50):
        pred = model(dummy_src, dummy_cam1, dummy_cam2)
        loss = criterion(pred, dummy_tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    no_fp16_time = time.time() - start
    no_fp16_speed = 50 * 16 / no_fp16_time
    
    print(f"      - Speed: {no_fp16_speed:.1f} images/s")
    print(f"      - Steps/s: {50 / no_fp16_time:.2f}")
    
    # Benchmark with FP16
    print("   Testing WITH Mixed Precision (FP16)...")
    scaler = torch.cuda.amp.GradScaler()
    
    start = time.time()
    for _ in range(50):
        with torch.cuda.amp.autocast():
            pred = model(dummy_src, dummy_cam1, dummy_cam2)
            loss = criterion(pred, dummy_tgt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    fp16_time = time.time() - start
    fp16_speed = 50 * 16 / fp16_time
    
    print(f"      - Speed: {fp16_speed:.1f} images/s")
    print(f"      - Steps/s: {50 / fp16_time:.2f}")
    
    print(f"\n   📈 FP16 Speedup: {fp16_speed / no_fp16_speed:.2f}x")
    
    # Memory usage
    print(f"\n   💾 GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   💾 GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 4. Summary
print("\n" + "=" * 60)
print("📋 Summary")
print("=" * 60)
print(f"\n   🖼️ Generation Speed: {gen_speed:.0f} images/s")
if torch.cuda.is_available():
    print(f"   🧠 Training Speed (FP16): {fp16_speed:.0f} images/s")
    print(f"   🧠 Training Speed (FP32): {no_fp16_speed:.0f} images/s")
    
    # Bottleneck analysis
    if gen_speed < fp16_speed:
        print(f"\n   ⚠️ BOTTLENECK: Data Generation is slower than Training!")
        print(f"      Recommendation: Increase num_workers or pre-generate data")
    else:
        print(f"\n   ✅ Training is the bottleneck (expected)")

print("\n" + "=" * 60)
