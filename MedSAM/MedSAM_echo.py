# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from skimage import io, transform
from echo_dataset import EchoDynamic
import argparse
from tqdm import tqdm
from typing import Any, Dict, List
import cv2


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None,)
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,)

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False,)  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(description="run inference on testing set based on MedSAM")
parser.add_argument("--video_path", type=str, default='/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic',
                    help="path to the data folder",)
parser.add_argument("--split", type=str, default='val', help="The dataset for inference.",)
parser.add_argument("--seg_path", type=str, default="assets/Echo_seg", help="path to the segmentation folder",)
parser.add_argument("--box", type=str, default='[25, 25, 84, 100]', help="bounding box of the segmentation target",)
parser.add_argument("--device", type=str, default="cuda:3", help="device")
parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth",
                    help="path to the trained model",)
args = parser.parse_args()


# box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
# transfer box_np t0 1024x1024 scale


def imgpreprocess(img):
    # %% image preprocessing
    img = np.transpose(img, (1, 2, 0))
    img_1024 = transform.resize(img, (1024, 1024), order=3, preserve_range=True,
                                anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8,
                                                     a_max=None)  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(args.device))

    return img_1024_tensor


def main():

    print("Loading model...")

    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(args.device)
    medsam_model.eval()

    dataset_kwargs = {"target_type": ['SmallFrame', 'LargeFrame'], "length": 32, "period": 3, }
    dataset = EchoDynamic(root=args.video_path, split=args.split, **dataset_kwargs)
    train_data = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    # dataset_val = EchoDynamic(root=args.video_path, split="val", **dataset_kwargs)
    # val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True)
    H, W = 112, 112
    box_np = np.array([[25, 25, 50, 70],
                       [50, 10, 90, 70],
                       [15, 55, 60, 100],
                       [50, 55, 90, 110]])
    num_box = 4

    for (img0, img1, middle, vid) in tqdm(train_data):
        print(f"Processing '{vid}'...")
        img0_1024 = imgpreprocess(img0.squeeze().numpy())
        img1_1024 = imgpreprocess(img1.squeeze().numpy())
        middle_1024 = imgpreprocess(middle.squeeze().numpy())

        for idx, box in enumerate(box_np):
            box = box[None, :]
            box_1024 = box / np.array([W, H, W, H]) * 1024

            with torch.no_grad():
                img0_embedding = medsam_model.image_encoder(img0_1024)  # (1, 256, 64, 64)
                img1_embedding = medsam_model.image_encoder(img1_1024)
                middle_embedding = medsam_model.image_encoder(middle_1024)

            img0_seg = medsam_inference(medsam_model, img0_embedding, box_1024, H, W)
            # vid_name = os.path.basename(vid[0])
            # io.imsave(join(args.seg_path, "seg_" + vid_name.split('.')[0] + '.png'),
            #           img0_seg, check_contrast=False, )
            img1_seg = medsam_inference(medsam_model, img1_embedding, box_1024, H, W)
            middle_seg = medsam_inference(medsam_model, middle_embedding, box_1024, H, W)

            # video_name = os.path.basename(vid[0])
            # save_base = os.path.join(args.seg_path, video_name)
            # os.makedirs(save_base, exist_ok=True)

            # cv2.imwrite(os.path.join(save_base, f"img0_{idx}.png"), img0_seg * 255)
            # cv2.imwrite(os.path.join(save_base, f"img1_{idx}.png"), img1_seg * 255)
            # cv2.imwrite(os.path.join(save_base, f"middle_{idx}.png"), middle_seg * 255)

    print("Done!")


if __name__ == "__main__":
    main()  # pragma: no cover

