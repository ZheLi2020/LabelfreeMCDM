"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import datetime
import time

import argparse
import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch

from utils.config_parser import init_config
# from utils.logger import init_logger

torch.backends.cudnn.benchmark = True
import numpy as np
from absl import logging
import pkg_resources

from datasets.get_dataset import get_dataset

from losses.get_loss import get_loss

from models.get_model import get_model

from trainer.get_trainer import get_trainer

# our internal file system; please comment out this line and change I/O to your own file system
# from utils.manifold_utils import MANIFOLD_BUCKET, MANIFOLD_PATH, pathmgr

from utils.torch_utils import init_seed
import wandb


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def main_ddp(rank, world_size, cfg):
    init_seed(cfg.seed)

    # set up distributed process groups
    os.environ["MASTER_ADDR"] = "10.76.21.14"
    os.environ["MASTER_PORT"] = find_free_port()

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda:%d" % rank)
    torch.cuda.set_device(device)
    logging.info(f"Use GPU {rank} ({torch.cuda.get_device_name(rank)}) for training")

    # prepare data
    train_sets, valid_sets, train_sets_epoches = get_dataset(cfg.data)
    if rank == 0:
        logging.info("train sets: " + ", ".join(["{} ({} samples)".format(ds.name, len(ds)) for ds in train_sets]))
        logging.info("val sets: " + ", ".join(["{} ({} samples)".format(ds.name, len(ds)) for ds in valid_sets]))

    train_sets_epoches = [np.inf if e == -1 else e for e in train_sets_epoches]

    train_loaders, valid_loaders = [], []
    for ds in train_sets:
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size // world_size,
                                                   num_workers=cfg.train.workers // world_size,
                                                   pin_memory=True, sampler=sampler,)
        train_loaders.append(train_loader)

    if rank == 0:
        # prepare tensorboard
        run = wandb.init(project="opticalflow", job_type="info", config=cfg, )

        # prepare validation dataset
        for ds in valid_sets:
            valid_loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True,
                                                       shuffle=False, drop_last=False,)
            valid_loaders.append(valid_loader)
        valid_size = sum([len(loader) for loader in valid_loaders])
        if cfg.train.valid_size == 0:
            cfg.train.valid_size = valid_size
        cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    else:
        valid_loaders = []

    # prepare model
    model = get_model(cfg.model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,)

    # prepare loss
    loss = get_loss(cfg.loss)

    # prepare training scipt
    trainer = get_trainer(cfg.trainer)(train_loaders, valid_loaders, model, loss, cfg.save_root,
                                       cfg.train, resume=cfg.resume, train_sets_epoches=train_sets_epoches,
                                       rank=rank, world_size=world_size,)

    trainer.train()

    torch.distributed.destroy_process_group()


def main_single(cfg, args):
    init_seed(cfg.seed)

    # set up distributed process groups
    # torch.cuda.set_device(device)

    # prepare data
    train_sets, valid_sets, train_sets_epoches = get_dataset(cfg.data)
    logging.info(f"Echo train sets: {train_sets.get_length()} samples")
    logging.info(f"Echo val sets: {valid_sets.get_length()} samples")

    train_loader = torch.utils.data.DataLoader(train_sets, batch_size=cfg.train.batch_size,
                                                   num_workers=cfg.train.workers, pin_memory=True)

    # prepare validation dataset
    valid_loader = torch.utils.data.DataLoader(valid_sets, batch_size=4, num_workers=4, pin_memory=True,
                                                   shuffle=False, drop_last=False,)

    valid_size = len(valid_loader)
    if cfg.train.valid_size == 0:
        cfg.train.valid_size = valid_size
    cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    # prepare model
    model = get_model(cfg.model).to(args.device)

    # prepare loss
    loss = get_loss(cfg.loss)

    # prepare training scipt
    trainer = get_trainer(cfg.trainer)(train_loader, valid_loader, model, loss, cfg.save_root, cfg.train, 'Echo',
                                       resume=cfg.resume, train_sets_epoches=train_sets_epoches, device=args.device,
                                       n_gpu=args.n_gpu)

    trainer.train_single()


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


def main(args):
    args.config = pkg_resources.resource_filename(__name__, args.config)

    # load config
    cfg = init_config(args.config)
    cfg.train.n_gpu = args.n_gpu

    run = wandb.init(project="opticalflow", job_type="info", config=cfg, )
    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)
    args.name = os.path.basename(args.config)[:-5]
    cfg.save_root = os.path.join(args.exp_folder, args.name[5:] + run_dir)
    if not os.path.exists(cfg.save_root):
        os.makedirs(cfg.save_root, exist_ok=True)

    set_logger(log_level='info', fname=os.path.join(cfg.save_root, 'output.log'))
    logging.info(f'args: {args}')
    logging.info(f'cfg: {cfg}')

    # DEBUG options
    cfg.train.DEBUG = args.DEBUG
    if args.DEBUG:
        cfg.data.update({"epoches_raw": 3,})
        cfg.train.update({"batch_size": 4, "epoch_num": 5, "epoch_size": 20, "print_freq": 1,
                          "record_freq": 1, "val_epoch_size": 2, "valid_size": 4, "save_iter": 2,})
        if "stage1" in cfg.train:
            cfg.train.stage1.update({"epoch": 5})
        if "stage2" in cfg.train:
            cfg.train.stage2.update({"epoch": 5})

    # pretrained model
    if args.model is not None:
        cfg.train.pretrained_model = args.model

    # init save_root: store files by curr_time
    if args.resume is not None:
        cfg.resume = True
        cfg.save_root = args.resume
    else:
        cfg.resume = False

        ## for the linux file system
        os.system("cp {} {}".format(args.config, os.path.join(cfg.save_root, "config.json")))
        if "base_configs" in cfg:
            os.system("cp {} {}".format(os.path.join(os.path.dirname(args.config), cfg.base_configs),
                                        os.path.join(cfg.save_root, cfg.base_configs),))

    logging.info("=> will save everything to {}".format(cfg.save_root))

    # show configurations
    logging.info(f"=> configurations \n {cfg} ")

    # spawn ddp
    # world_size = args.n_gpu
    # torch.multiprocessing.spawn(main_ddp, args=(world_size, cfg), nprocs=world_size,)
    main_single(cfg, args)

    logging.info("Completed!")
    return


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/echo_aug+hg+mf.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--exp_folder", default="results")
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--DEBUG", action="store_true")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
