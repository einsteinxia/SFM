import os.path
import random

import kornia
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from augmentation import AugBasic
from config import RandomRealsConfig
from data.dataloaders import get_dataset
from distillation.eval import Evaluator
from distillation.eval_inherit import Evaluator as InheritEvaluator
from models import get_model


def eval_random_reals(cfg: RandomRealsConfig):

    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    save_dir = os.path.join(
        "baselines",
        "results",
        cfg.method,
        cfg.dataset,
        f"ipc_{cfg.ipc}",
        cfg.model if cfg.eval_mode == "normal" else f"{cfg.model}_{cfg.eval_mode}_{cfg.ci_eval_model}",
        f"seed_{cfg.random_seed}",
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}.pth".format(cfg.model))
    checkpoint_path = os.path.join(save_dir, "checkpoint_{}.pth".format(cfg.model))

    if os.path.exists(save_path) and cfg.skip_if_exists:
        print(
            "This baseline has already been run. Set `--skip_if_exists=False` to force a redo."
        )
        exit(0)

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode="center",
        data_root=cfg.data_root,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=cfg.workers_per_gpu,
        batch_size=cfg.batch_size_per_gpu * torch.cuda.device_count(),
    )

    random_real_images = train_dataset.get_random_reals(ipc=cfg.ipc).cuda()
    random_real_labels = torch.cat(
        [torch.tensor([c] * cfg.ipc) for c in range(train_dataset.num_classes)]
    ).cuda()

    ds = TensorDataset(
        random_real_images.detach().clone(), random_real_labels.detach().clone()
    )

    loader = DataLoader(ds, batch_size=min(100, len(random_real_images)), shuffle=True)

    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()

    if cfg.eval_mode == "CI":
        if cfg.ci_eval_model is None:
            raise ValueError("`--ci_eval_model` is required when `--eval_mode CI` is used.")
        if cfg.classifier_path is None:
            raise ValueError("`--classifier_path` is required when `--eval_mode CI` is used.")

        distill_model, distill_num_feats = get_model(
            cfg.model, distributed=torch.cuda.device_count() > 1
        )
        eval_model, eval_num_feats = get_model(
            cfg.ci_eval_model, distributed=torch.cuda.device_count() > 1
        )

        evaluator = InheritEvaluator(
            train_loader=loader,
            test_loader=test_loader,
            distill_model=distill_model,
            eval_model=eval_model,
            eval_model_name=cfg.ci_eval_model,
            checkpoint_path=checkpoint_path,
            projector_path=None,
            classifier_path=cfg.classifier_path,
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
            random_seed=cfg.random_seed,
            eval_mode=cfg.eval_mode,
        )
    else:
        eval_model, num_feats = get_model(
            cfg.model, distributed=torch.cuda.device_count() > 1
        )

        evaluator = Evaluator(
            train_loader=loader,
            test_loader=test_loader,
            model=eval_model,
            checkpoint_path=checkpoint_path,
            augmentor=augmentor,
            epochs=cfg.eval_epochs,
            eval_it=cfg.eval_it,
            patience=cfg.patience,
            checkpoint_it=cfg.checkpoint_it,
            normalize=train_dataset.normalize,
            num_feats=num_feats,
            num_classes=train_dataset.num_classes,
            num_eval=cfg.num_eval,
            random_seed=cfg.random_seed,
        )

    evaluator.train_and_eval()

    top1_mean = np.mean(evaluator.top1_list)
    top1_std = np.std(evaluator.top1_list)

    top5_mean = np.mean(evaluator.top5_list)
    top5_std = np.std(evaluator.top5_list)

    save_dict = {
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top5_mean": top5_mean,
        "top5_std": top5_std,
    }
    
    torch.save(save_dict, save_path)
    print(f"Results saved to {save_path}")
    os.remove(checkpoint_path)

    log_file = os.path.join(os.path.dirname(save_path), "evaluation_results.txt")

    with open(log_file, "a") as f:  
        f.write("Evaluation Method: {}\n".format(cfg.method))
        f.write("Evaluation Dataset: {}\n".format(cfg.dataset))
        f.write("Evaluation IPC: {}\n".format(cfg.ipc))
        f.write("Evaluation Times: {}\n".format(cfg.num_eval))
        f.write("Evaluation Model: {}\n".format(cfg.model))
        f.write("Evaluation Eval Mode: {}\n".format(cfg.eval_mode))
        if cfg.ci_eval_model is not None:
            f.write("Evaluation CI Eval Model: {}\n".format(cfg.ci_eval_model))
        f.write("Top 1 Mean ± Std: {:.2f} ± {:.2f}\n".format(top1_mean * 100, top1_std * 100))
        f.write("Top 5 Mean ± Std: {:.2f} ± {:.2f}\n".format(top5_mean * 100, top5_std * 100))

    print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean * 100, top1_std * 100))
    print("Top 5 Mean ± Std: {:.2f} ± {:.2f}".format(top5_mean * 100, top5_std * 100))

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = RandomRealsConfig(explicit_bool=True).parse_args()
    print(args)
    eval_random_reals(args)
