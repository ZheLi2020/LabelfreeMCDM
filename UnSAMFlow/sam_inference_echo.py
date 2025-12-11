"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from https://github.com/facebookresearch/segment-anything/blob/main/scripts/amg.py
"""

import argparse
import glob
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from typing import Any, Dict, List

import cv2  # type: ignore

# from utils.manifold_utils import pathmgr
import numpy as np

from segment_anything.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
from datasets.echo_dataset import EchoDynamicOne
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description=("Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."))

# parser.add_argument(
#     "--input",
#     type=str,
#     required=True,
#     help="Path to either a single input image or folder of images.",
# )

parser.add_argument("--dataset", type=str, default='Echo', help="The dataset for inference.",)
parser.add_argument("--split", type=str, default='train', help="The dataset for inference.",)
parser.add_argument("--video_path", type=str, default='/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic', help="The dataset for inference.",)
parser.add_argument("--seg_path", type=str, default='/vol/idea_longterm/ot70igyn/MedSAM/assets/Echo_medseg', help="The dataset for inference.",)
parser.add_argument("--device", type=str, default="cuda:3", help="The device to run generation on.")

parser.add_argument("--output", type=str, default='./datasets/Echo_obj',
                    help=("Path to the directory where masks will be output. Output will be either a folder "
                          "of PNGs per image or a single json with COCO-style masks."),)

parser.add_argument("--model_type", type=str, default='vit_h', help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",)

parser.add_argument("--checkpoint", type=str, default='./segment_anything/sam_vit_h_4b8939.pth',
                    help="The path to the SAM checkpoint to use for mask generation. medsam_vit_b, sam_vit_h_4b8939",)

parser.add_argument("--convert-to-rle", action="store_true", help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."),)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument("--points-per-side", type=int, default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",)

amg_settings.add_argument("--points-per-batch", type=int, default=None,
    help="How many input points to process simultaneously in one batch.",)

amg_settings.add_argument("--pred-iou-thresh", type=float, default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",)

amg_settings.add_argument("--stability-score-thresh", type=float, default=None,
                          help="Exclude masks with a stability score lower than this threshold.",)

amg_settings.add_argument("--stability-score-offset", type=float, default=None,
    help="Larger values perturb the mask more when measuring stability score.",)

amg_settings.add_argument("--box-nms-thresh", type=float, default=None,
    help="The overlap threshold for excluding a duplicate mask.",)

amg_settings.add_argument("--crop-n-layers", type=int, default=None,
    help=("If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."),)

amg_settings.add_argument("--crop-nms-thresh", type=float, default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",)

amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None,
    help="Larger numbers mean image crops will overlap more.",)

amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",)

amg_settings.add_argument("--min-mask-region-area", type=int, default=None,
    help=("Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."),)


def write_masks_to_folder(masks: List[Dict[str, Any]], name: str, path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{name}_{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [str(i), str(mask_data["area"]),
                         *[str(x) for x in mask_data["bbox"]],
                         *[str(x) for x in mask_data["point_coords"][0]],
                         str(mask_data["predicted_iou"]),
                         str(mask_data["stability_score"]),
                         *[str(x) for x in mask_data["crop_box"]],]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    dataset_kwargs = {"target_type": ['SmallFrame', 'LargeFrame'], "length": 32, "period": 3, }
    dataset = EchoDynamicOne(root=args.video_path, split=args.split, **dataset_kwargs)
    train_data = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    # dataset_val = EchoDynamic(root=args.video_path, split="val", **dataset_kwargs)
    # val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True)

    for (img0, img1, middle, video_path) in tqdm(train_data):
        print(f"Processing '{video_path}'...")
        img0 = np.transpose(img0.squeeze().numpy(), (1, 2, 0))
        img1 = np.transpose(img1.squeeze().numpy(), (1, 2, 0))
        middle = np.transpose(middle.squeeze().numpy(), (1, 2, 0))
        masks_img0 = generator.generate(img0)
        masks_img1 = generator.generate(img1)
        masks_mid = generator.generate(middle)

        video_name = os.path.basename(video_path[0])
        save_base = os.path.join(args.output, video_name)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks_img0, 'img0', save_base)
            write_masks_to_folder(masks_img1, 'img1', save_base)
            write_masks_to_folder(masks_mid, 'middle', save_base)
        else:
            save_file0 = save_base + "img0.json"
            with open(save_file0, "w") as f:
                json.dump(masks_img0, f)

            save_file1 = save_base + "img1.json"
            with open(save_file1, "w") as f:
                json.dump(masks_img1, f)

            save_file_m = save_base + "mid.json"
            with open(save_file_m, "w") as f:
                json.dump(masks_mid, f)
    print("Done!")


def main_mask_to_full_seg(args):

    import imageio.v2 as imageio

    vid_list = os.listdir(args.seg_path)
    img_list = ['img0', 'img1', 'middle']
    num_obj = 4

    for vid_name in tqdm(vid_list):
        for frame in img_list:
            masks = []
            for idx in range(num_obj):
                img_name = os.path.join(args.seg_path, vid_name, f'{frame}_{idx}.png')
                m = imageio.imread(img_name)/255
                masks.append(m.astype(np.uint8))

            masks = np.array(masks)

            H, W = masks.shape[1:]
            masks_area = np.array([np.sum(mask) for mask in masks])

            # drop mask if it equals the full frame
            masks_clean = masks[masks_area < H * W]
            masks_area = masks_area[masks_area < H * W]

            # sort the class ids by area, largest to smallest
            area_order = np.argsort(masks_area)[::-1]
            masks_area = masks_area[area_order]
            masks_clean = masks_clean[area_order]

            # add a "background mask" for pixels that are not included in any masks
            masks_clean_aug = np.concatenate((np.ones((1, H, W)), masks_clean), axis=0)
            masks_area_aug = np.array([H * W] + masks_area.tolist())
            masks_area_aug = np.array(masks_area_aug, dtype=np.float32)

            unified_mask = np.argmin(masks_clean_aug * masks_area_aug[:, None, None]
                                     + (1 - masks_clean_aug) * (H * W + 1), axis=0,)

            unique_classes = np.unique(unified_mask)
            mapping = np.zeros((unique_classes.max() + 1))
            for i, cl in enumerate(unique_classes):
                mapping[cl] = i
            new_mask = mapping[unified_mask]

            if new_mask.max() > 255:  # almost not existent
                print("More than 256 masks detect for image {}".format(frame))
                new_mask[new_mask > 255] = 0
            new_mask = new_mask.astype(np.uint8)

            save_path = os.path.join(args.output, vid_name.split('.')[0] + f'_{frame}.png')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, new_mask)


def main_mask_to_key_objects(args):
    import imageio.v2 as imageio
    vid_list = os.listdir(args.seg_path)
    img_list = ['img0', 'img1', 'middle']
    num_obj = 4

    for vid_name in tqdm(vid_list):
        for frame in img_list:
            masks = []
            for idx in range(num_obj):
                img_name = os.path.join(args.seg_path, vid_name, f'{frame}_{idx}.png')
                m = imageio.imread(img_name) / 255
                masks.append(m.astype(np.uint8))

            masks = np.array(masks)

            H, W = masks.shape[1:]
            obj_masks = np.zeros((0, H, W), dtype=np.uint8)

            for mask_id in range(len(masks)):
                mask = masks[mask_id, :, :]

                num_unique_masks = ((masks * mask[None, :, :]).sum((1, 2)) > 0).sum()
                obj_masks = np.concatenate((obj_masks, (mask[None, :, :] * 255).astype(np.uint8)), axis=0)

            save_path = os.path.join(args.output, vid_name.split('.')[0] + f'_{frame}.npy')
            np.save(save_path, obj_masks)


def invoke_main() -> None:
    args = parser.parse_args()
    # main(args)

    # main_mask_to_full_seg(args)

    main_mask_to_key_objects(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
