<div align="center">

# 🎯 U-Net Novel View Synthesis

### *Generate New 3D Viewpoints from Single Images Using Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange.svg)](https://gradio.app/)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900?style=for-the-badge&logo=nvidia" alt="CUDA"/>
</p>

---

**A deep learning project that learns to synthesize novel viewpoints of 3D objects from a single input image, leveraging the power of U-Net architecture with perceptual loss optimization.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Web Demo](#-web-demo)
- [Results](#-results)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Novel View Synthesis (NVS) is a fundamental problem in computer vision and graphics. This project implements a **U-Net based encoder-decoder architecture** that learns to generate new viewing angles of 3D geometric shapes from a single input image.

### Key Highlights:
- 🎨 **9 Geometric Shapes**: Cube, Sphere, Cylinder, Cone, Pyramid, Torus, Octahedron, Dodecahedron, Icosahedron
- 📐 **Fibonacci Sphere Sampling**: Optimal uniform distribution of camera viewpoints
- 🚀 **Real-time Inference**: Fast prediction with GPU acceleration
- 🌐 **Interactive Web Demo**: Gradio-powered interface for easy testing

---

## ✨ Features

| Feature                      | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| **U-Net Architecture**       | Encoder-decoder with skip connections for preserving spatial details |
| **Perceptual Loss**          | VGG-based feature matching for high-quality image generation         |
| **Mixed Precision Training** | FP16 training for faster computation and reduced memory              |
| **Auto Resume**              | Automatic checkpoint saving and training resumption                  |
| **Live Dashboard**           | Real-time training metrics visualization                             |
| **Web Interface**            | Interactive Gradio demo for testing predictions                      |
| **Fibonacci Sampling**       | Mathematically optimal viewpoint distribution                        |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    U-Net Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Image (256×256×3) + Target Camera (3)               │
│                        ↓                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              ENCODER (Contracting Path)              │   │
│   │  Conv→BN→ReLU→Conv→BN→ReLU→MaxPool (×4 blocks)      │   │
│   │  64 → 128 → 256 → 512 channels                       │   │
│   └─────────────────────────────────────────────────────┘   │
│                        ↓                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   BOTTLENECK                         │   │
│   │              1024 channels (16×16)                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                        ↓                                     │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              DECODER (Expanding Path)                │   │
│   │  UpConv→Concat→Conv→BN→ReLU (×4 blocks)             │   │
│   │  512 → 256 → 128 → 64 channels                       │   │
│   │  + Skip Connections from Encoder                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                        ↓                                     │
│              Output Image (256×256×3)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/Mo-ra778/UNet-Novel-View-Synth.git
cd UNet-Novel-View-Synth

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
Pillow>=9.0.0
gradio>=3.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

---

## 🚀 Usage

### Quick Start - Web Demo

Launch the interactive web demo to test the model:

```bash
cd "Phase4_UNet_Pro - نسخة/Phase4_UNet_Pro"
python web_demo.py
```

Then open your browser at `http://localhost:7860`

### Command Line Testing

```bash
python test_model.py --input path/to/image.png --elevation 45 --azimuth 90
```

### Full Evaluation

```bash
python full_evaluation.py
```

---

## 📁 Project Structure

```
UNet-Novel-View-Synth/
│
├── 📂 Phase4_UNet_Pro - نسخة/
│   └── 📂 Phase4_UNet_Pro/
│       │
│       ├── 🧠 model_unet.py          # U-Net architecture definition
│       ├── 📊 dataloader_phase4.py   # Data loading and preprocessing
│       ├── 🎯 loss_perceptual.py     # Perceptual loss implementation
│       ├── 🏋️ train_phase4_unet.py   # Training script
│       ├── 🧪 test_model.py          # Model testing utility
│       ├── 📈 full_evaluation.py     # Comprehensive evaluation
│       ├── 🌐 web_demo.py            # Gradio web interface
│       ├── 🔧 benchmark.py           # Performance benchmarking
│       │
│       ├── 📂 checkpoints/           # Saved model weights
│       ├── 📂 training_samples/      # Training visualizations
│       └── 📂 evaluation_results/    # Evaluation outputs
│
├── 📄 .gitignore                     # Git ignore rules
├── 📄 README.md                      # This file
└── 📄 requirements.txt               # Python dependencies
```

---

## 🏋️ Training

### Start Training from Scratch

```bash
python train_phase4_unet.py
```

### Training Configuration

Edit the training script to customize:

| Parameter       | Default | Description                   |
| --------------- | ------- | ----------------------------- |
| `batch_size`    | 8       | Batch size for training       |
| `learning_rate` | 1e-4    | Initial learning rate         |
| `num_epochs`    | 100     | Total training epochs         |
| `image_size`    | 256     | Input/output image resolution |
| `num_shapes`    | 9       | Number of geometric shapes    |

### Training Features

- ✅ **Automatic Checkpointing**: Saves best model based on validation loss
- ✅ **Learning Rate Scheduling**: Reduces LR on plateau
- ✅ **Mixed Precision (FP16)**: Faster training with reduced memory
- ✅ **Live Dashboard**: Real-time loss and PSNR visualization
- ✅ **Resume Capability**: Automatically resumes from last checkpoint

---

## 🌐 Web Demo

The project includes an interactive **Gradio** web interface:

### Features:
- 🖼️ **Image Upload**: Upload source images or select from examples
- 🎚️ **Camera Controls**: Adjust elevation (0-90°) and azimuth (0-360°)
- ⚡ **Real-time Prediction**: Instant novel view generation
- 📊 **Quality Metrics**: Display PSNR and inference time

### Launch Demo:

```bash
python web_demo.py
```

Access at: `http://localhost:7860`

---

## 📊 Results

### Performance Metrics

| Metric             | Value       |
| ------------------ | ----------- |
| **PSNR**           | ~25-30 dB   |
| **SSIM**           | ~0.85-0.92  |
| **Inference Time** | ~50ms (GPU) |
| **Model Size**     | ~124 MB     |

### Supported Shapes

| Shape       | Preview | Shape        | Preview |
| ----------- | ------- | ------------ | ------- |
| Cube        | 🟦       | Sphere       | 🔵       |
| Cylinder    | 🔷       | Cone         | 🔺       |
| Pyramid     | 🔻       | Torus        | 🍩       |
| Octahedron  | 💎       | Dodecahedron | ⬡       |
| Icosahedron | ⚽       |              |         |

---

## ⚙️ Configuration

### Camera Distribution

The project uses **Fibonacci Sphere Sampling** for optimal camera placement:

```
Fibonacci Spiral Distribution:
• 40 viewpoints per shape
• Uniform coverage of viewing hemisphere  
• Elevation: 0° to 90°
• Azimuth: 0° to 360°
```

This ensures:
- ✅ Maximum diversity in training data
- ✅ No clustering at poles
- ✅ Mathematically optimal distribution

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution:
- 🎨 Add more 3D shapes
- 🚀 Optimize inference speed
- 📱 Mobile-friendly demo
- 📚 Improve documentation
- 🧪 Add more test cases

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Mo-ra778**

- GitHub: [@Mo-ra778](https://github.com/Mo-ra778)

---

## 🙏 Acknowledgments

- PyTorch team for the amazing deep learning framework
- Gradio for the easy-to-use web interface library
- The computer vision research community

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Made with ❤️ and PyTorch**

</div>
