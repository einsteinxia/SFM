
import torch
from tap import Tap
from typing import Literal

class StatisticCfg(Tap):
    dataset: str = "imagenet-100"
    model: str = "mocov3_vitb"
    data_root: str = "/root/datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    augs_per_batch: int = 10
    real_res: int = 256
    crop_res: int = 224

    ipc: int = 1

    mode: Literal["random", "center", "closest", "augment"] = "center"

    device_count: int = torch.cuda.device_count()

    skip_if_exists: bool = True

    job_tag: str | None = None
    job_id: str | None = None
