import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm
from Trainer import Model
import config as cfg
from echo_one import EchoNetOne
from absl import logging
from benchmark.utils.pytorch_msssim import ssim_matlab
import pickle

exp = os.path.abspath('.').split('/')[-1]


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5


def inference(batch_size, video_path, device):
    device = torch.device(device)
    saved_ep = 5
    split = 'VAL'

    run_dir = '20250805-233810'
    save_path = os.path.join('log/train_EchoNet', run_dir)
    set_logger(log_level='info', fname=os.path.join(save_path, f'generate_real{saved_ep}_{split}.log'))
    logging.info(f'Save path is {save_path}')

    # cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    # cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])

    model = Model(device=args.device, max_levels=args.pyramid_levels)
    model.load_model(save_path, device=device, name=str(saved_ep))
    model.eval()
    model.device(args.device)

    dataset_kwargs = {"target_type": ['SmallFrame', 'LargeFrame'], "length": 32, "period": 3, }
    # dataset_train = EchoNetTwo(root=video_path, split="train", **dataset_kwargs)
    # dataset_train.set_return_key(True)

    # train_data = DataLoader(dataset_train, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    # args.step_per_epoch = train_data.__len__()

    dataset_val = EchoNetOne(root=video_path, split=split, **dataset_kwargs)
    dataset_val.set_return_key(True)
    # val_data = DataLoader(dataset_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    # dataset_test = EchoNetTwo(root=video_path, split="test", **dataset_kwargs)
    # dataset_test.set_return_key(True)
    # test_data = DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    logging.info(f'Generating motion features for {split} dataset from epoch {saved_ep}.')
    time_stamp = time.time()

    # for dataset in [dataset_train, dataset_val, dataset_test]:
    save_path = os.path.join(save_path, f'motion_{saved_ep}', split)
    os.makedirs(save_path, exist_ok=True)
    for dataset in [dataset_val]:
        psnr_list, ssim_list = [], []
        for idx in tqdm(range(dataset.get_length())):

            img0, gt, img1, imgs_path = dataset.get_frames(idx, return_name=True)

            img0 = torch.from_numpy(img0.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.
            img1 = torch.from_numpy(img1.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.
            gt = torch.from_numpy(gt.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.

            pred, af, mf = model.inference(img0, img1, TTA=True, fast_TTA=True)

            mf1 = torch.mean(torch.mean(mf[-2], dim=-1), dim=-1)
            mf2 = torch.mean(torch.mean(mf[-1], dim=-1), dim=-1)
            motion_feature = torch.cat([mf1, mf2], dim=1).view(-1).detach().to('cpu')

            imgs_path = imgs_path.split('/')[-1].split('.')[0]
            torch.save(motion_feature, os.path.join(save_path, f"{imgs_path}.pt"))

            ssim = 0
            for j in range(gt.shape[0]):
                ssim = ssim_matlab(gt[j].unsqueeze(0), torch.round(pred[j].unsqueeze(0) * 255) / 255.).detach().cpu().numpy()
            out = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            out = np.round(out * 255) / 255.
            gt = gt.cpu().numpy()
            gt = gt.transpose(0, 2, 3, 1)
            psnr = []
            for j in range(gt.shape[0]):
                psnr.append(-10 * math.log10(((gt - out) * (gt - out)).mean()))
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        mean_psnr = np.mean(np.array(psnr_list))
        mean_ssim = np.mean(np.array(ssim_list))
        logging.info(f"Avg PSNR: {mean_psnr} SSIM: {mean_ssim}")
        #torch.save(motion_features, os.path.join(save_path, f"{data_split[split_idx]}_features.pt"))

        time_interval = time.time() - time_stamp
        time_stamp = time.time()
        logging.info(f'time: {time_interval:.2f}')


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--device', type=str, default='cuda:1', help='model')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--pyramid_levels', default=4, type=int, help='batch size')
    parser.add_argument('--video_path', default='/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic', type=str, help='data path of ucf101')
    # parser.add_argument('--syn_path', default='/vol/ideadata/at70emic/projects/reproductions/EchoNet-Synthetic/samples/synthetic/lvdm_dynamic_motion_50k', type=str, help='frames path')
    args = parser.parse_args()

    os.makedirs('log', exist_ok=True)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    inference(args.batch_size, args.video_path, args.device)
        
