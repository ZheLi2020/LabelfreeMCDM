# LabelfreeMCDM
The repository is for our paper Label-free Motion-Conditioned Diffusion Model for Cardiac Ultrasound Synthesis [![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2512.09418) <!-- Update with actual arXiv ID when available -->

## 🏗️ Architecture

The framework consists of three main components:

1.  **Motion-Appearance Feature Extractor (MAFE):** A network that explicitly disentangles motion and appearance.
2.  **Pseudo Ground Truth Supervision:**
    * **ReID Model:** Generates pseudo appearance embeddings to guide appearance learning ($\mathcal{L}_{reid}$).
    * **UnSAMFlow:** Generates pseudo optical flow fields to refine motion features ($\mathcal{L}_{flow}$).
3.  **Label-Free Diffusion:** A Motion-Conditioned Diffusion Model (MCDM) conditioned on the precomputed, self-supervised motion features.

## 🚀 Installation

### Prerequisites
* Linux
* Python 3.10
* PyTorch 2.5
* NVIDIA GPU (Tested on A100s for training)


## 💾 Dataset Preparation
This project uses the **EchoNet-Dynamic** dataset. The processing follows EchoNet-Synthetic (https://github.com/HReynaud/EchoNet-Synthetic).

    

# Coming soon

