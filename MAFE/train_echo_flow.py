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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from echo_one import EchoNetFlow
from absl import logging

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


def train(model, args):

    device = torch.device(args.device)
    step = 0
    nr_eval = 0
    best_epoch = 0
    best_psnr = 0

    run_dir = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join('log/train_EchoNet', run_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    set_logger(log_level='info', fname=os.path.join(save_path, 'output.log'))
    logging.info(f'Save path is {save_path}')
    logging.info(f'config: {args}')
    writer = SummaryWriter(save_path)

    dataset_kwargs = {"target_type": ['SmallFrame', 'LargeFrame'], "length": 32, "period": 3, }
    dataset = EchoNetFlow(root=args.video_path, reid_path=args.reid_path, flow_path=args.flow_path,
                          split="train", return_type=args.method_type, **dataset_kwargs)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, drop_last=False)

    args.step_per_epoch = train_data.__len__()

    dataset_val = EchoNetFlow(root=args.video_path, reid_path=args.reid_path, flow_path=args.flow_path,
                              split="val", **dataset_kwargs)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=0)

    logging.info('Train echo motion only with small frame and large frame, no next frame.')
    time_stamp = time.time()

    total_epoch = args.total_epoch
    loss_reid, loss_flow = 0, 0
    for epoch in range(total_epoch):
        # sampler.set_epoch(epoch)
        for i, data in tqdm(enumerate(train_data)):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = data['imgs'].to(device, non_blocking=True) / 255.
            img0, img1, gt = imgs[:, 0, ...], imgs[:, 1, ...], imgs[:, 2, ...]
            imgs_all = torch.cat([img0, img1], dim=1).to(device)
            learning_rate = get_learning_rate(step)

            if args.method_type == 'flow':
                flow_data = data['flow']
                flow_data = flow_data.to(device)
                _, loss, loss_flow = model.update_flow(imgs_all, gt, flow_data, args.flow_ratio, learning_rate, training=True,
                                            device=device)

            if args.method_type == 'flownorm':
                flow_data = data['flow']
                flow_data = flow_data.to(device)
                _, loss, loss_flow = model.update_flow_norm(imgs_all, gt, flow_data, args.flow_ratio, learning_rate, training=True,
                                            device=device)
            if args.method_type == 'both':
                reid_features = data['reid']
                flow_data = data['flow']
                reid_features, flow_data = reid_features.to(args.device), flow_data.to(args.device)
                _, loss, loss_reid, loss_flow = model.update_both(imgs_all, gt, reid_features, flow_data,
                                                                  args.flow_ratio, args.reid_ratio, learning_rate,
                                                                  training=True, device=device)
            if args.method_type == 'bothnorm':
                reid_features = data['reid']
                flow_data = data['flow']
                reid_features, flow_data = reid_features.to(args.device), flow_data.to(args.device)
                _, loss, loss_reid, loss_flow = model.update_both_norm(imgs_all, gt, reid_features, flow_data,
                                                                  args.flow_ratio, args.reid_ratio, learning_rate,
                                                                  training=True, device=device)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1:  # and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
                logging.info(f'Train epoch:{epoch} {i}/{args.step_per_epoch} time:{data_time_interval:.2f}+{train_time_interval:.2f} '
                             f'loss:{loss:.5f} reid loss:{loss_reid:.5f} flow loss:{loss_flow:.5f}')
            step += 1
            # break
        nr_eval += 1
        if nr_eval % 3 == 0:
            psnr = evaluate(model, val_loader, nr_eval, args.method_type, device)
            if psnr > best_psnr:
                best_epoch = epoch
                best_psnr = psnr
                model.save_model(epoch, save_path)

    model.save_model(total_epoch, save_path)

    logging.info(f'The best epoch is {best_epoch}, the best psnr is {best_psnr}')


def evaluate(model, val_loader, nr_eval, method_type, device):
    psnr = []
    for data in tqdm(val_loader):
        imgs = data['imgs'].to(device, non_blocking=True) / 255.

        img0, img1, gt = imgs[:, 0, ...], imgs[:, 1, ...], imgs[:, 2, ...]
        imgs_all = torch.cat([img0, img1], dim=1).to(device)

        with torch.no_grad():
            if method_type == 'flow':
                flow_data = data['flow']
                flow_data = flow_data.to(device)
                pred, _ = model.update_flow(imgs_all, gt, flow_data, training=False)
            if method_type == 'flownorm':
                flow_data = data['flow']
                flow_data = flow_data.to(device)
                pred, _ = model.update_flow_norm(imgs_all, gt, flow_data, training=False)
            if method_type == 'both':
                reid_features = data['reid']
                flow_data = data['flow']
                reid_features, flow_data = reid_features.to(args.device), flow_data.to(args.device)
                pred, _ = model.update_both(imgs_all, gt, reid_features, flow_data, training=False)
            if method_type == 'bothnorm':
                reid_features = data['reid']
                flow_data = data['flow']
                reid_features, flow_data = reid_features.to(args.device), flow_data.to(args.device)
                pred, _ = model.update_both_norm(imgs_all, gt, reid_features, flow_data, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    psnr = np.array(psnr).mean()
    logging.info(f'Evaluate at num eval: {nr_eval}, psnr: {psnr}')

    return psnr


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--device', type=str, default='cuda:0', help='model')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--total_epoch', default=100, type=int, help='batch size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--pyramid_levels', default=4, type=int, help='batch size')
    parser.add_argument('--flow_ratio', default=0.01, type=float, help='batch size')
    parser.add_argument('--reid_ratio', default=1, type=float, help='batch size')
    parser.add_argument('--video_path', default='/vol/ideadata/at70emic/projects/EchoSynExt/datasets/EchoNet-Dynamic', type=str, help='data path of ucf101')
    # parser.add_argument('--data_path', default='./data/vimeo_triplet', type=str, help='frames path')
    parser.add_argument('--reid_path', default='/vol/idea_longterm/ot70igyn/EchoNet-Synthetic/external/reindentification/saveddata/20250603-180230/reid_10000',
                        type=str, help='data path of echo reid feature')
    parser.add_argument('--flow_path',
                        default='/vol/idea_longterm/ot70igyn/UnSAMFlow/results/aug+hg+mf20250731-100335-solar-bird-54/pred_flow',
                        type=str, help='data path of echo flow')
    parser.add_argument('--method_type', default='bothnorm', type=str, help='use which pseudo gt. choices [reid, flow, both]')
    args = parser.parse_args()
    # torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    # torch.cuda.set_device(args.local_rank)
    # if args.local_rank == 0 and not os.path.exists('log'):
    os.makedirs('log', exist_ok=True)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    model = Model(device=args.device, max_levels=args.pyramid_levels)
    train(model, args)
        
