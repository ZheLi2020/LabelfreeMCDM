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
This project uses the **EchoNet-Dynamic** dataset. The processing follows [EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic).


## 🏃 Training

The training process is divided into two stages: feature extraction and diffusion modeling.

### Stage 1: Train MAFE (Motion-Appearance Feature Extractor)

Train the feature extractor to learn disentangled motion features.

```bash
cd MAFE
python train_echo_flow.py
```

*This stage utilizes pseudo-labels from a pre-trained ReID model and UnSAMFlow.*

### Stage 2: Train MCDM (Diffusion Model)

Train the latent video diffusion model conditioned on the extracted motion features.
* **Hardware:** 2x NVIDIA A100 GPUs (at least 2 GPUs, better 4 GPUs)

```bash
accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 echosyn/lvdm/train_motion.py --config echosyn/lvdm/configs/default_uncondition.yaml
```

## 📜 Citation

If you find this code or paper useful for your research, please cite:

```bibtex
@article{li2025label,
  title={Label-free Motion-Conditioned Diffusion Model for Cardiac Ultrasound Synthesis},
  author={Li, Zhe and Reynaud, Hadrien and M{\"u}ller, Johanna P and Kainz, Bernhard},
  journal={The International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD)},
  year={2025}
}
```

## 🔗 Related Repositories

This work builds upon and relates to the following projects:

* **EchoNet-Synthetic:** [https://github.com/HReynaud/EchoNet-Synthetic](https://github.com/HReynaud/EchoNet-Synthetic)
* **EMA-VFI:** [https://github.com/MCG-NJU/EMA-VFI](https://github.com/MCG-NJU/EMA-VFI)
* **UnSAMFlow:** [https://github.com/facebookresearch/UnSAMFlow](https://github.com/facebookresearch/UnSAMFlow)
* **MedSAM:** [https://github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)


