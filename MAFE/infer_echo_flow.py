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
from echo_one import EchoNetFlow
from absl import logging
import imageio.v2 as imageio
from benchmark.utils.pytorch_msssim import ssim_matlab

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


def save_flow(pred_flows, save_path, vid_name):
    H = 112
    flow = pred_flows.detach().cpu().numpy().transpose([0, 2, 3, 1])
    for i in range(flow.shape[0]):
        flow_png = vid_name + f'_{i}.png'

        mag, ang = cv2.cartToPolar(flow[i][..., 0], flow[i][..., 1])
        hsv = np.zeros([H, H, 3]).astype(np.float32)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        min_value = np.min(flow_img)
        flow_img = flow_img - min_value
        max_value = np.max(flow_img)
        flow_img = 255 * flow_img / max_value
        imageio.imwrite(os.path.join(save_path, flow_png), flow_img.astype(np.uint8))


def inference(video_path, device):
    device = torch.device(device)
    saved_ep = 86
    split = 'TRAIN'
    args.method_type = 'flow'

    run_dir = '20250811-090946'
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
    # dataset_train = EchoNetFlow(root=video_path, split="train", **dataset_kwargs)
    # dataset_train.set_return_key(True)

    # train_data = DataLoader(dataset_train, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    # args.step_per_epoch = train_data.__len__()

    dataset_val = EchoNetFlow(root=video_path, reid_path=args.reid_path, flow_path=args.flow_path,
                              split=split, return_type=args.method_type, **dataset_kwargs)
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
    # flow_path = os.path.join(save_path, f'flow_{saved_ep}', split)
    # os.makedirs(flow_path, exist_ok=True)
    for dataset in [dataset_val]:
        psnr_list, ssim_list = [], []
        for idx in tqdm(range(dataset.get_length())):

            img0, gt, img1, _, _, imgs_path = dataset.get_frames(idx)

            img0 = torch.from_numpy(img0.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.
            img1 = torch.from_numpy(img1.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.
            gt = torch.from_numpy(gt.copy()).unsqueeze(0).to(device, non_blocking=True) / 255.

            flow_pred, pred, af, mf = model.inference(img0, img1, TTA=True, fast_TTA=True)

            mf1 = torch.mean(torch.mean(mf[-2], dim=-1), dim=-1)
            mf2 = torch.mean(torch.mean(mf[-1], dim=-1), dim=-1)
            motion_feature = torch.cat([mf1, mf2], dim=1).view(-1).detach().to('cpu')

            imgs_path = imgs_path.split('/')[-1].split('.')[0]
            torch.save(motion_feature, os.path.join(save_path, f"{imgs_path}.pt"))

            # save_flow(flow_pred, flow_path, imgs_path)

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
        std_psnr = np.std(np.array(psnr_list))
        mean_ssim = np.mean(np.array(ssim_list))
        std_ssim = np.std(np.array(ssim_list))
        logging.info(f"Avg PSNR: {mean_psnr} Std PSNR: {std_psnr}")
        logging.info(f"Avg SSIM: {mean_ssim} Std SSIM: {std_ssim}")
        #torch.save(motion_features, os.path.join(save_path, f"{data_split[split_idx]}_features.pt"))

        time_interval = time.time() - time_stamp
        time_stamp = time.time()
        logging.info(f'time: {time_interval:.2f}')


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--device', type=str, default='cuda:3', help='model')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--pyramid_levels', default=4, type=int, help='batch size')
    parser.add_argument('--video_path', default='/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic', type=str, help='data path of ucf101')
    # parser.add_argument('--syn_path', default='/vol/ideadata/at70emic/projects/reproductions/EchoNet-Synthetic/samples/synthetic/lvdm_dynamic_motion_50k', type=str, help='frames path')
    parser.add_argument('--reid_path',
                        default='/vol/idea_longterm/ot70igyn/EchoNet-Synthetic/external/reindentification/saveddata/20250603-180230/reid_10000',
                        type=str, help='data path of echo reid feature')
    parser.add_argument('--flow_path',
                        default='/vol/idea_longterm/ot70igyn/UnSAMFlow/results/aug+hg+mf20250731-100335-solar-bird-54/pred_flow',
                        type=str, help='data path of echo flow')
    parser.add_argument('--method_type', default='flow', type=str,
                        help='use which pseudo gt. choices [reid, flow, both]')
    args = parser.parse_args()

    os.makedirs('log', exist_ok=True)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    inference(args.video_path, args.device)
        
