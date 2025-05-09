# 📸 SVD-Based Image Processing & Compression

A comprehensive Python project demonstrating the power of **Singular Value Decomposition (SVD)** in image processing—featuring **image compression**, **denoising**, and **eigenface generation**.

Built with **NumPy**, **OpenCV**, **Matplotlib**, **Scikit-learn**, and more.

---

## 🚀 Features

### 🔹 Image Compression
- Compress images using a selected number of singular values.
- Supports **grayscale** and **color** formats.
- Visual quality evaluation via **PSNR** and **SSIM**.
- Shows **compression ratio** and **storage savings**.

### 🔹 Interactive SVD Explorer
- Interactive widgets to explore SVD compression in real-time.
- Adjust number of components `k` and view live reconstructions.
- Visual + metric comparisons.

### 🔹 Image Denoising
- Adds **Gaussian noise** and denoises using SVD.
- Comparison with **Gaussian blur**.
- Evaluate denoising quality visually and quantitatively.

### 🔹 Eigenfaces (Face Recognition)
- Uses the **Olivetti Faces** dataset from Scikit-learn.
- Visualizes **mean face** and top **eigenfaces**.
- Reconstruct faces using eigenface components.

---

## 📦 Requirements

- Python 3.7+
- Required Python libraries:
  ```bash
  numpy
  matplotlib
  opencv-python
  scikit-image
  scikit-learn
  ipywidgets
  IPython
  ```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushp2563/svd-image-model.git
   cd svd-image-model
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### 🔸 Basic Image Compression
```python
from main import load_image, svd_compress_reconstruct

img = load_image("test_image.png")
compressed = svd_compress_reconstruct(img, k=50)
```

---

### 🔸 Interactive SVD Explorer

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Run inside a notebook cell:
   ```python
   from main import run_interactive_svd_explorer
   run_interactive_svd_explorer('test_image.png')
   ```

---

### 🔸 Image Denoising
```python
from main import add_gaussian_noise, svd_denoise

noisy = add_gaussian_noise(image, sigma=25)
denoised = svd_denoise(noisy, k_ratio=0.1)
```

---

### 🔸 Eigenfaces Demo
```python
from main import run_eigenfaces_demo

run_eigenfaces_demo(n_components=100, n_faces_to_show=10)
```

---

## 📁 Project Structure

```
svd-image-processing/
├── main.ipynb          # Main implementation notebook
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── test_image.png      # Sample test image
```

---

## 📊 Example Outputs

### 📉 Image Compression
- Original vs. compressed image comparisons
- PSNR & SSIM quality metrics
- Compression ratio estimation

### 🧭 Interactive Explorer
- Real-time SVD reconstruction
- Dynamic metric updates

### 🧹 Denoising
- Comparison between clean, noisy, and denoised images
- Visual and numerical analysis

### 🧠 Eigenfaces
- Mean face visualization
- Top eigenfaces gallery
- Reconstructed faces using principal components

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request 🚀

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Scikit-learn](https://scikit-learn.org/) for the Olivetti Faces dataset
- Contributors to NumPy, OpenCV, Matplotlib, and scikit-image
- The Python open-source community ❤️
