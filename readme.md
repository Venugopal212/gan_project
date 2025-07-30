# GAN Project – 2D Synthetic Data & PathMNIST Generation

This project showcases two parts of generative adversarial network (GAN) implementations:

- **Part 1:** Training GANs on synthetic 2D datasets (sine wave and spiral)
- **Part 2:** Conditional and Deep Convolutional GANs on the PathMNIST dataset from [MedMNIST](https://medmnist.com/)

🔗 **GitHub Repo:** [https://github.com/Venugopal212/gan_project](https://github.com/Venugopal212/gan_project)

---

## 🔧 Environment Setup

Activate your virtual environment:

```bash
.venv\Scripts\activate  # On Windows
# OR
source .venv/bin/activate  # On macOS/Linux
````

## 📁 Part 1: 2D GANs on Synthetic Data

Navigate to the directory:

```bash
cd part1_2d_gans
```

### ➤ Step 1: Train on Sine Wave

```bash
python train_sine.py
```

### ➤ Step 2: Train on Spiral Data

After `train_sine.py` completes:

```bash
python train_spiral.py
```

---

## 📁 Part 2: PathMNIST GANs

Navigate to the second project directory:

```bash
cd ..
cd part2_pathmnist
```

### ➤ Visualize Conditional GAN (cGAN)

```bash
python visualize.py --type cgan --ckpt logs/cgan/ --out cgan_outputs/
```

### ➤ Visualize Deep Convolutional GAN (DCGAN)

```bash
python visualize.py --type dcgan --ckpt logs/dcgan/ --out dcgan_output.png
```

---

## 📦 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install torch torchvision matplotlib numpy tqdm scikit-learn seaborn medmnist
```

---