"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import logging
import os

from abc import abstractmethod
from absl import logging
import numpy as np
import torch
import wandb

# from utils.manifold_utils import pathmgr
from utils.torch_utils import (AdamW, bias_parameters, load_checkpoint, other_parameters,
                               save_checkpoint, weight_parameters,)

from .object_cache import ObjectCache


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loaders, valid_loaders, model, loss_func, save_root, config, data_name,
                 resume=False, train_sets_epoches=None, rank=0, world_size=1, device=None, n_gpu=1):
        self.cfg = config
        self.save_root = save_root
        self.train_loaders, self.valid_loaders = train_loaders, valid_loaders
        self.train_sets_epoches = train_sets_epoches
        self.data_name = data_name

        self.rank, self.world_size = rank, world_size
        self.device = device
        self.n_gpu = n_gpu
        self.loss_func = loss_func
        self.best_loss = 100000

        if resume:  # load all states
            self._load_resume_ckpt(model)
        else:
            self.model = self._init_model(model)
            self.i_epoch, self.i_iter = 0, 0
            self.i_train_set = 0

            self.optimizer = self._create_optimizer()
            if self.data_name == 'Echo':
                self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches)
            else:
                self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches[self.i_train_set])

            self.best_error = np.inf

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _run_one_epoch_echo(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def log(self, s):
        if self.rank == 0:
            logging.info(s)

    def set_up_obj_cache(self, cache_size=500):
        self.obj_cache = ObjectCache(cache_size=cache_size)

    def train(self):

        if (self.cfg.pretrained_model is not None):  # if using a pretrained model, evaluate that first to compare
            if self.rank == 0:
                self._validate_with_gt()
            torch.distributed.barrier()

        for _epoch in range(self.i_epoch, self.cfg.epoch_num):
            self._run_one_epoch()

            if self.i_epoch >= sum(self.train_sets_epoches[: (self.i_train_set + 1)]):
                self.i_train_set += 1
                self.optimizer = (self._create_optimizer())  # reset the states of optimizer as well
                self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches[self.i_train_set])

            if self.rank == 0:
                if self.i_epoch % self.cfg.val_epoch_size == 0:
                    self._validate_with_gt()
                    self.log(" * Epoch {} validation complete.".format(self.i_epoch))

            torch.distributed.barrier()

    def train_single(self):

        if self.cfg.pretrained_model is not None:  # if using a pretrained model, evaluate that first to compare
            self._validate_with_gt()

        update_th = [int(self.train_sets_epoches / 2), self.train_sets_epoches]
        for _epoch in range(self.i_epoch, self.cfg.epoch_num):
            self._run_one_epoch_echo()

            if self.i_epoch in update_th:
                self.optimizer = (self._create_optimizer())  # reset the states of optimizer as well
                self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches)

            if self.i_epoch % 50 == 0:
                self.save_model(name="model_ep{}".format(self.i_epoch))

        self._validate_with_gt()
        logging.info(" * Epoch {} validation complete.".format(self.i_epoch))

    def validate(self):
        self._validate_with_gt()

    def _init_model(self, model):
        model = model.to(self.device)

        if self.cfg.pretrained_model:
            self.log("=> using pre-trained weights {}.".format(self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)
            model.module.load_state_dict(weights)
        elif self.n_gpu > 1:
            self.log("=> Train from scratch.")
            model.module.init_weights()
        else:
            self.log("=> Train from scratch.")
            model.init_weights()

        self.log("number of parameters: {}".format(self.count_parameters(model)))
        self.log("gpu memory allocated (model parameters only): {} Bytes".format(torch.cuda.memory_allocated()))
        return model

    def _create_optimizer(self):
        self.log("=> setting {} optimizer".format(self.cfg.optim))
        if self.n_gpu > 1:
            param_groups = [{"params": bias_parameters(self.model.module),
                             "weight_decay": self.cfg.bias_decay,},
                            {"params": weight_parameters(self.model.module),
                             "weight_decay": self.cfg.weight_decay,},
                            {"params": other_parameters(self.model.module), "weight_decay": 0},]
        else:
            param_groups = [{"params": bias_parameters(self.model),
                             "weight_decay": self.cfg.bias_decay, },
                            {"params": weight_parameters(self.model),
                             "weight_decay": self.cfg.weight_decay, },
                            {"params": other_parameters(self.model), "weight_decay": 0}, ]

        if self.cfg.optim == "adamw":
            optimizer = AdamW(param_groups, self.cfg.lr, betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == "adam":
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta), eps=1e-7,)
        else:
            raise NotImplementedError(self.cfg.optim)

        return optimizer

    def _create_scheduler(self, optimizer, epoches=np.inf):

        if self.i_train_set < 1:  # try only the last loader uses onecyclelr
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
            return scheduler

        if "lr_scheduler" in self.cfg.keys():
            self.log("=> setting {} scheduler".format(self.cfg.lr_scheduler.module))

            params = self.cfg.lr_scheduler.params

            if self.cfg.lr_scheduler.module == "OneCycleLR":
                params["epochs"] = min(epoches, self.cfg.epoch_num - self.i_epoch)
                params["steps_per_epoch"] = self.cfg.epoch_size

            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.module)(optimizer, **params)
        else:  # a dummy scheduler by default
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

        return scheduler

    def _load_resume_ckpt(self, model):
        self.log("==> resuming")

        with open(os.path.join(self.save_root, "model_best139_ckpt.pth.tar"), "rb") as f:
            ckpt_dict = torch.load(f)

        if "iter" not in ckpt_dict.keys():
            ckpt_dict["iter"] = ckpt_dict["epoch"] * self.cfg.epoch_size
        if "best_error" not in ckpt_dict.keys():
            ckpt_dict["best_error"] = np.inf
        self.i_epoch, self.i_iter, self.best_error = (ckpt_dict["epoch"], ckpt_dict["iter"],
                                                      ckpt_dict["best_error"],)
        # self.i_train_set = np.where(self.i_epoch < np.cumsum(self.train_sets_epoches))[0][0]
        self.i_train_set = 0

        self.model = model.to(self.device)
        self.model.load_state_dict(ckpt_dict["state_dict"])
        # self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer, self.train_sets_epoches)

        if "optimizer_dict" in ckpt_dict.keys():
            self.optimizer.load_state_dict(ckpt_dict["optimizer_dict"])
        if "scheduler_dict" in ckpt_dict.keys():
            self.scheduler.load_state_dict(ckpt_dict["scheduler_dict"])

        return

    # def _prepare_device(self, n_gpu_use):
    #     """
    #     setup GPU device if available, move model into configured device
    #     """
    #     n_gpu = torch.cuda.device_count()
    #     if n_gpu_use > 0 and n_gpu == 0:
    #         self.log(
    #             "Warning: There's no GPU available on this machine,"
    #             "training will be performed on CPU."
    #         )
    #         n_gpu_use = 0
    #     if n_gpu_use > n_gpu:
    #         self.log(
    #             "Warning: The number of GPU's configured to use is {}, "
    #             "but only {} are available.".format(n_gpu_use, n_gpu)
    #         )
    #         n_gpu_use = n_gpu
    #     device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    #     list_ids = list(range(n_gpu_use))
    #     self.log("=> gpu in use: {} gpu(s)".format(n_gpu_use))
    #     self.log(
    #         "device names: {}".format([torch.cuda.get_device_name(i) for i in list_ids])
    #     )
    #     return device, list_ids

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, name, save_with_runtime=True):
        if save_with_runtime:
            if self.n_gpu > 1:
                models = {"epoch": self.i_epoch, "iter": self.i_iter, "best_error": self.best_error,
                          "state_dict": self.model.module.state_dict(),
                          "optimizer_dict": self.optimizer.state_dict(),
                          "scheduler_dict": self.scheduler.state_dict(),}
            else:
                models = {"epoch": self.i_epoch, "iter": self.i_iter, "best_error": self.best_error,
                          "state_dict": self.model.state_dict(),
                          "optimizer_dict": self.optimizer.state_dict(),
                          "scheduler_dict": self.scheduler.state_dict(), }
        else:
            if self.n_gpu > 1:
                models = {"state_dict": self.model.module.state_dict()}
            else:
                models = {"state_dict": self.model.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best=False)
