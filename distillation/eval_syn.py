import copy
import glob
import multiprocessing
import os
import random
import shutil
import signal
import sys
import types
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from augmentation import AugBasic
from config import EvalCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model, LinearClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        distill_model: nn.Module,
        eval_model: nn.Module,
        eval_model_name: str,
        checkpoint_path: str | None,
        projector_path: str | None,
        classifier_path: str | None,
        augmentor: nn.Module,
        epochs: int,
        eval_it: int,
        patience: int,
        checkpoint_it: int,
        normalize: Callable[[Tensor], Tensor],
        distill_num_feats: int,
        eval_num_feats: int,
        num_classes: int,
        num_eval: int,
        random_seed: int = 0,
        T: float = 4.0,
        alpha: float = 0.5,
        eval_mode: str = "inherit",
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.distill_model = distill_model
        self.eval_model = eval_model
        self.eval_model_name = eval_model_name
        self.checkpoint_path = checkpoint_path
        self.projector_path = projector_path
        self.classifier_path = classifier_path
        self.augmentor = augmentor
        self.epochs = epochs
        self.eval_it = eval_it
        self.patience = patience
        self.checkpoint_it = checkpoint_it
        self.normalize = normalize
        self.distill_num_feats = distill_num_feats
        self.eval_num_feats = eval_num_feats    
        self.num_classes = num_classes
        self.num_eval = num_eval
        self.random_seed = random_seed
        self.T = T
        self.alpha = alpha
        self.eval_mode = eval_mode

        # self.best_fc = get_fc(
        #     num_feats=self.num_feats,
        #     num_classes=self.num_classes,
        #     distributed=torch.cuda.device_count() > 1,
        # )
        # self.best_fc = None

        self.top1_list = []
        self.top5_list = []

        self.reset()

        # this MUST be after self.reset()
        # self.load_checkpoint()

    def reset(self):

        torch.manual_seed(self.random_seed + len(self.top1_list))
        random.seed(self.random_seed + len(self.top1_list))
        np.random.seed(self.random_seed + len(self.top1_list))


        if self.eval_mode == "CI":
            self.projector = nn.Linear(self.eval_num_feats, self.distill_num_feats).cuda()
            # self.projector = nn.Sequential(
            #     nn.Linear(self.eval_num_feats, 1024), 
            #     nn.Linear(1024, 1024), 
            #     nn.Linear(1024, 1024), 
            #     nn.Linear(1024, self.distill_num_feats), 
            # ).cuda()

            self.fc = get_fc(
                num_feats=self.distill_num_feats,
                num_classes=self.num_classes,
                distributed=torch.cuda.device_count() > 1,
            )
            save_info = torch.load(self.classifier_path, weights_only=False)
            top1_mean = save_info["top1_mean"]
            top1_std = save_info["top1_std"]
            top5_mean = save_info["top5_mean"]
            top5_std = save_info["top5_std"]
            best_fc_state_dict = save_info["best_fc"]
            print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean * 100, top1_std * 100))
            print("Top 5 Mean ± Std: {:.2f} ± {:.2f}".format(top5_mean * 100, top5_std * 100))

            self.fc.linear.load_state_dict(best_fc_state_dict)
            self.fc.linear.requires_grad_(False)
            self.projector.requires_grad_(True)

            self.optimizer = torch.optim.Adam(
                list(self.fc.parameters()) + list(self.projector.parameters()),
                0.01
                * (self.train_loader.batch_size / torch.cuda.device_count())
                / 256.0,  # linear scaling rule
                weight_decay=0,  # we do not apply weight decay
            )

        # self.projector = nn.Linear(self.eval_num_feats, self.distill_num_feats).cuda()
        elif self.eval_mode == "KD":
            self.distill_fc = get_fc(
                num_feats=self.distill_num_feats,
                num_classes=self.num_classes,
                distributed=torch.cuda.device_count() > 1,
            )
            save_info = torch.load(self.classifier_path, weights_only=False)
            top1_mean = save_info["top1_mean"]
            top1_std = save_info["top1_std"]
            top5_mean = save_info["top5_mean"]
            top5_std = save_info["top5_std"]
            best_fc_state_dict = save_info["best_fc"]
            print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean * 100, top1_std * 100))
            print("Top 5 Mean ± Std: {:.2f} ± {:.2f}".format(top5_mean * 100, top5_std * 100))

            self.distill_fc.linear.load_state_dict(best_fc_state_dict)
            self.distill_fc.linear.requires_grad_(False)

            self.fc = get_fc(
                num_feats=self.eval_num_feats,
                num_classes=self.num_classes,
                distributed=torch.cuda.device_count() > 1,
            )
            self.optimizer = torch.optim.Adam(
                list(self.fc.parameters()),
                0.001
                * (self.train_loader.batch_size / torch.cuda.device_count())
                / 256.0,  # linear scaling rule
                weight_decay=0,  # we do not apply weight decay
            )
        else:   # normal training   --hard labels
            # self.projector = nn.Linear(self.eval_num_feats, self.eval_num_feats).cuda()
            self.fc = get_fc(
                num_feats=self.eval_num_feats,
                num_classes=self.num_classes,
                distributed=torch.cuda.device_count() > 1,
            )

            self.optimizer = torch.optim.Adam(
                list(self.fc.parameters()),
                0.001
                * (self.train_loader.batch_size / torch.cuda.device_count())
                / 256.0,  # linear scaling rule
                weight_decay=0,  # we do not apply weight decay
            )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs, eta_min=0
        )

        self.current_epoch = 0
        self.patience_counter = 0
        self.top1_best = 0
        self.top5_best = 0
        self.scaler = GradScaler()

    def train_and_eval(self):

        while len(self.top1_list) < self.num_eval:

            for e in tqdm(
                range(self.current_epoch, self.epochs),
                desc="Training Linear Head",
                leave=True,
                initial=self.current_epoch,
                total=self.epochs,
            ):
                self.current_epoch = e

                if self.current_epoch % self.checkpoint_it == 0:
                    self.save_checkpoint()

                self.train_one_epoch()

                if (
                    self.current_epoch % self.eval_it == 0 and self.eval_it != -1
                ) or self.current_epoch == self.epochs - 1:
                    top1, top5 = self.evaluate()
                    print("Top1: {:.2f}".format(top1 * 100))
                    if top1 <= self.top1_best:
                        self.patience_counter += 1
                        print("Losing patience: {}".format(self.patience_counter))
                        if self.patience_counter == self.patience:
                            print("Out of patience! Stopping training.")
                            break

                    else:
                        self.best_fc = copy.deepcopy(self.fc)
                        # self.best_projector = copy.deepcopy(self.projector)
                        self.patience_counter = 0
                        self.top1_best = top1
                        self.top5_best = top5

            self.top1_list.append(self.top1_best)
            self.top5_list.append(self.top5_best)
            self.reset()

        # torch.save(self.best_projector.state_dict(), "projector/{}_projector.pth".format(self.eval_model_name))
        print("Finished!")
        print(self.top1_list)

    def train_one_epoch(self):

        for x, y in tqdm(self.train_loader, desc="Epoch Progress", leave=False):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with autocast():

                with torch.no_grad():
                    x = self.augmentor(x)
                    x = self.normalize(x)
                    z = self.eval_model(x)
                    g = self.distill_model(x)

                if self.eval_mode == "CI":
                    p = self.projector(z)

                    # p_norm = F.normalize(p, p=2, dim=1)  # [100, 768]
                    # g_norm = F.normalize(g, p=2, dim=1)  # [100, 768]
                    # loss = F.mse_loss(p_norm, g_norm)

                    # mask_ratio = 0.1
                    # mask = torch.rand_like(z).to(z.device)
                    # mask = (mask > mask_ratio).float()  
    
                    # # 2. mask
                    # mask_z = z * mask
                    # p = self.projector(mask_z)

                    loss = F.mse_loss(p, g)

                elif self.eval_mode in ["JT", "ST"]:
                    p = self.projector(z)
                    out = self.fc(p)
                    loss = nn.functional.cross_entropy(out, y)

                elif self.eval_mode == "KD":
                    logits = self.distill_fc(g)
                    soft_labels = F.softmax(logits / self.T, dim=1)
                    out = self.fc(z)
                    stu_probs = F.log_softmax(out / self.T, dim=1)
                    kl_loss = F.kl_div(stu_probs, soft_labels, reduction="batchmean") * (self.T ** 2)
                    ce_loss = nn.functional.cross_entropy(out, y)
                    loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

                    # print('kl_loss.item()', kl_loss.item())
                    # print('ce_loss.item()', ce_loss.item())
                else:
                    out = self.fc(z)
                    loss = nn.functional.cross_entropy(out, y)
                
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            self.scaler.update()

        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self):
        num_classes = self.test_loader.dataset.num_classes
        top1_metric = MulticlassAccuracy(
            average="micro", num_classes=num_classes, top_k=1
        ).cuda()
        if self.test_loader.dataset.num_classes >= 5:
            top5_metric = MulticlassAccuracy(
                average="micro", num_classes=num_classes, top_k=5
            ).cuda()

        for x, y in tqdm(self.test_loader, desc="Evaluating Linear Head", leave=False):
            x = x.cuda()
            y = y.cuda()
            x = self.normalize(x)
            z = self.eval_model(x)
 
            if self.eval_mode in ["CI", "JT", "ST"]:
                p = self.projector(z)
                out = self.fc(p)
            else:
                out = self.fc(z)

            top1_metric.update(out, y)

            if self.test_loader.dataset.num_classes >= 5:
                top5_metric.update(out, y)

        top1 = top1_metric.compute().item()
        if self.test_loader.dataset.num_classes >= 5:
            top5 = top5_metric.compute().item()
        else:
            top5 = 0.0

        return top1, top5

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            return
        print("Saving checkpoint...")
        random_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        save_dict = {
            "fc": self.fc.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "top1_best": self.top1_best,
            "top5_best": self.top5_best,
            "patience_counter": self.patience_counter,
            "random_state": random_state,
            "top1_list": self.top1_list,
            "top5_list": self.top5_list,
        }

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(save_dict, self.checkpoint_path+".tmp")
        os.rename(
            self.checkpoint_path+".tmp", self.checkpoint_path
        )
        print("Saved!")

    def load_checkpoint(self):
        if self.checkpoint_path is None:
            return
        if os.path.exists(self.checkpoint_path):
            print("Checkpoint found! Resuming...")
            load_dict = torch.load(self.checkpoint_path, weights_only=False)
            self.fc.load_state_dict(load_dict["fc"])
            self.optimizer.load_state_dict(load_dict["optimizer"])
            self.scheduler.load_state_dict(load_dict["scheduler"])
            self.current_epoch = load_dict["current_epoch"]

            self.top1_best = load_dict["top1_best"]
            self.top5_best = load_dict["top5_best"]
            self.patience_counter = load_dict["patience_counter"]

            torch.set_rng_state(load_dict["random_state"]["torch"])
            torch.cuda.set_rng_state_all(load_dict["random_state"]["cuda"])
            random.setstate(load_dict["random_state"]["python"])
            np.random.set_state(load_dict["random_state"]["numpy"])

            self.top1_list = load_dict["top1_list"]
            self.top5_list = load_dict["top5_list"]

        else:
            print("No checkpoint found, starting from scratch...")


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy("file_system")

    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)

    cfg = EvalCfg(explicit_bool=True).parse_args()

    model_dir = os.path.join("logged_files", cfg.method, cfg.job_tag, cfg.dataset, cfg.model)
    # model_dir = '/home/joey_tr/xqx/linear-gradient-matching/src/baselines/results/real_centroids/imagenet-100/dinov2_vitb'
    print("Searching for saved data in {}".format(model_dir))
    syn_set_files = sorted(
        list(glob.glob(os.path.join(model_dir, "**", "data.pth"), recursive=True))
    )
    if len(syn_set_files) == 0:
        print(f"No data found at {model_dir}.")
        print("Exiting...")
        exit()
    run_dir = "/".join(syn_set_files[0].split("/")[:-1])

    save_dir = os.path.join(run_dir, "eval")
    save_file = os.path.join(save_dir, "{}.pth".format(cfg.eval_model))
    checkpoint_path = os.path.join(save_dir, "checkpoint_{}.pth".format(cfg.model))
    # if os.path.exists(save_file) and cfg.skip_if_exists:
    #     print("This eval already done.")
    #     print("Exiting...")
    #     exit()

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=cfg.num_workers,
        batch_size=cfg.real_batch_size,
    )

    if len(list(syn_set_files)) > 1:
        print("Warning: multiple syn sets found. Using the first one.")

    syn_set = torch.load(syn_set_files[0], weights_only=False)
    syn_images = syn_set["images"].cuda()
    
    syn_labels = syn_set["labels"].cuda()

    
    print('syn_images.shape', syn_images.shape)
    print('syn_labels', syn_labels)
    print('syn_images[0]', syn_images[0])
    
    print("loaded file from ", run_dir)
    print("eval model is ", cfg.eval_model)

    distill_model, distill_num_feats = get_model(
        cfg.model, distributed=torch.cuda.device_count() > 1
    )

    eval_model, eval_num_feats = get_model(
        cfg.eval_model, distributed=torch.cuda.device_count() > 1
    )

    ds = TensorDataset(syn_images.detach().clone(), syn_labels.detach().clone())

    loader = DataLoader(ds, batch_size=min(100, len(syn_images)), shuffle=True)


    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()
    augmentor = augmentor.cuda()

    evaluator = Evaluator(
        train_loader=loader,
        test_loader=test_loader,
        distill_model=distill_model,
        eval_model=eval_model,
        eval_model_name=cfg.eval_model,
        checkpoint_path=checkpoint_path,
        projector_path=cfg.projector_path,
        augmentor=augmentor,
        epochs=cfg.eval_epochs,
        eval_it=cfg.eval_it,
        patience=cfg.patience,
        checkpoint_it=cfg.checkpoint_it,
        normalize=train_dataset.normalize,
        distill_num_feats=distill_num_feats,
        eval_num_feats=eval_num_feats,
        num_classes=train_dataset.num_classes,
        num_eval=cfg.num_eval,
        classifier_path=cfg.classifier_path,
        T = cfg.T,
        alpha = cfg.alpha,
        eval_mode=cfg.eval_mode,
    )

    evaluator.train_and_eval()

    top1_mean = float(np.mean(evaluator.top1_list))
    top1_std = float(np.std(evaluator.top1_list))

    top5_mean = float(np.mean(evaluator.top5_list))
    top5_std = float(np.std(evaluator.top5_list))

    save_dict = {
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top5_mean": top5_mean,
        "top5_std": top5_std,
        "best_fc": evaluator.best_fc.linear.state_dict(),
    }

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results saved to {save_file}")
    torch.save(obj=save_dict, f=save_file)
    os.remove(checkpoint_path)

    log_file = os.path.join(save_dir,  f"ipc_{cfg.ipc}_method_{cfg.method}_evaluation_results.txt")

    with open(log_file, "a") as f:  
        f.write(f"distillation model: {cfg.model}\n")
        f.write(f"eval model: {cfg.eval_model}\n")
        f.write(f"ipc: {cfg.ipc}\n")
        f.write(f"method: {cfg.method}\n")
        f.write("Top 1 Mean ± Std: {:.2f} ± {:.2f}\n".format(top1_mean * 100, top1_std * 100))
        f.write("Top 5 Mean ± Std: {:.2f} ± {:.2f}\n".format(top5_mean * 100, top5_std * 100))

    print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean * 100, top1_std * 100))
    print("Top 5 Mean ± Std: {:.2f} ± {:.2f}".format(top5_mean * 100, top5_std * 100))