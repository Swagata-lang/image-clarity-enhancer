# 🚀 Image Clarity Enhancer

An AI-powered web application that enhances low-resolution images using **ESRGAN (Enhanced Super Resolution Generative Adversarial Network)**.

The application allows users to upload blurry or low-resolution images and generates a high-quality enhanced version with improved sharpness, details, and resolution.

---

# 📌 Features

- 🖼️ Upload JPG, JPEG and PNG images
- 🤖 AI-powered image super resolution using ESRGAN
- 📈 Improves image clarity and sharpness
- 📥 Download enhanced image
- ⚡ Fast Flask backend
- 🌐 Responsive web interface
- 🔒 Secure file validation
- 🎯 Supports GPU acceleration (CUDA) if available

---

# 🏗 Project Architecture

```
Image-Clarity-Enhancer
│
├── app.py
├── requirements.txt
├── templates
│   ├── index.html
│   └── upload.html
│
├── static
│   ├── uploads
│   └── outputs
│
├── experiments
│   └── pretrained_models
│       └── ESRGAN
│            └── ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
│
├── basicsr
│
└── README.md
```

---

# 🧠 AI Model

This project uses

**ESRGAN_SRx4_DF2KOST**

which is trained on

- DF2K Dataset
- OST Dataset

The model performs **4× Super Resolution**, reconstructing high-frequency image details while preserving textures.

Model File

```
ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
```

---

# 💻 Tech Stack

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap

### Backend

- Python
- Flask
- Flask-CORS

### AI

- PyTorch
- BasicSR
- ESRGAN

### Image Processing

- OpenCV
- NumPy

---

# 📋 Requirements

## Operating System

Recommended

- Ubuntu 22.04 LTS
- WSL2 Ubuntu (Windows)

Supported

- Linux ✅
- WSL2 ✅
- Docker ✅

Windows Native

⚠ Not officially recommended by BasicSR developers due to dependency issues.

---

# Python Version

Python

```
3.8
```

Recommended

---

# CUDA (Optional)

CUDA compatible GPU is recommended for faster inference.

Without CUDA, CPU inference is supported but significantly slower.

---

# Installation Guide

## Step 1

Clone the repository

```bash
git clone https://github.com/<your-username>/Image-Clarity-Enhancer.git

cd Image-Clarity-Enhancer
```

---

## Step 2

Create Conda Environment

```bash
conda create -n basicsr python=3.8

conda activate basicsr
```

---

## Step 3

Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Step 4

Install PyTorch

### CPU

```bash
pip install torch torchvision torchaudio
```

### CUDA

Refer to

https://pytorch.org/get-started/locally/

and install the appropriate CUDA version.

---

## Step 5

Install Project Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is unavailable

```bash
pip install flask
pip install flask-cors
pip install opencv-python
pip install numpy
pip install pillow
pip install tqdm
pip install pyyaml
pip install lmdb
pip install scipy
pip install requests
```

---

## Step 6

Install BasicSR

Clone

```bash
git clone https://github.com/XPixelGroup/BasicSR.git
```

Install

```bash
cd BasicSR

python setup.py develop
```

Return

```bash
cd ..
```

---

## Step 7

Download ESRGAN Model

Download

```
ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
```

Place inside

```
experiments/
    pretrained_models/
        ESRGAN/
```

Result

```
experiments
└── pretrained_models
    └── ESRGAN
        └── ESRGAN_SRx4_DF2KOST_official-ff704c30.pth
```

---

## Step 8

Create Required Directories

```
static
│
├── uploads
└── outputs
```

If missing

```bash
mkdir -p static/uploads

mkdir -p static/outputs
```

---

## Step 9

Run Flask Server

```bash
python app.py
```

Output

```
Running on

http://127.0.0.1:5000
```

---

## Step 10

Open Browser

Visit

```
http://127.0.0.1:5000
```

or

```
http://127.0.0.1:5000/upload-form
```

Upload an image and click

```
Upload & Enhance
```

Wait for inference.

The enhanced image will appear with a download option.

---

# Supported Image Formats

✔ PNG

✔ JPG

✔ JPEG

Maximum upload size

```
16 MB
```

---

# Project Workflow

```
User

↓

Uploads Image

↓

Flask Backend

↓

Save Image

↓

Load ESRGAN Model

↓

AI Enhancement

↓

Save Enhanced Image

↓

Return Image URL

↓

Display Enhanced Image

↓

Download Result
```

---

# API Endpoint

## Upload Image

```
POST

/upload
```

Form Data

```
file
```

Returns

```json
{
    "original": "/static/uploads/image.jpg",
    "enhanced": "/static/outputs/enhanced_image.jpg"
}
```

---

# Configuration

Inside

```python
app.config.update(...)
```

| Variable | Description |
|------------|------------------------------|
| UPLOAD_FOLDER | Stores uploaded images |
| OUTPUT_FOLDER | Stores enhanced images |
| MAX_CONTENT_LENGTH | Maximum upload size |
| ALLOWED_EXTENSIONS | Allowed image formats |
| MODEL_PATH | ESRGAN model location |

---

# Future Improvements

- Drag & Drop Upload
- Multiple AI Models
- Face Enhancement
- Batch Processing
- Image Comparison Slider
- Image History
- User Authentication
- Cloud Storage
- Docker Deployment
- Kubernetes Deployment
- REST API
- Mobile Responsive UI

---

# Known Issues

- Windows native installation may fail because BasicSR is primarily developed and tested on Linux.
- Large images may require significant RAM or GPU VRAM.
- CUDA is recommended for faster inference.

---

# License

This project uses the ESRGAN and BasicSR frameworks under their respective open-source licenses.

---

# Acknowledgements

- XPixelGroup (BasicSR)
- ESRGAN Authors
- PyTorch
- Flask
- OpenCV

---

## Author

Developed by

**Swagata**

ECE Undergraduate

AI • Computer Vision • Full Stack Development

```
"Enhancing pixels, one image at a time."
```
