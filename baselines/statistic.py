import os

import kornia
import torch
from torch import Tensor, nn
from tqdm import tqdm
from models import get_fc, get_model
from config import StatisticCfg
from data.dataloaders import (
    BaseRealDataset,
    get_dataset,
)
from models import get_model


from sklearn.cluster import KMeans
import torch

def get_kmeans_centers(soft_labels: torch.Tensor, num_clusters: int = 10) -> torch.Tensor:
    """
    Perform k-means clustering on soft labels and return cluster centers.

    Args:
        soft_labels (torch.Tensor): Input tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
        num_clusters (int): Number of clusters to form.

    Returns:
        torch.Tensor: Tensor of shape (num_clusters, D) representing the cluster centers.
    """
    # Convert the tensor to a numpy array for k-means
    soft_labels_np = soft_labels.cpu().numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(soft_labels_np)

    # Extract cluster centers and convert back to a tensor
    centers = torch.tensor(kmeans.cluster_centers_, dtype=soft_labels.dtype, device=soft_labels.device)

    return centers

@torch.no_grad()
def get_cluster_logits(
    labels: Tensor,
    model: nn.Module,
    train_dataset: BaseRealDataset,
    ipc: int,
) -> Tensor:
    crop = kornia.augmentation.CenterCrop(224)
    logits_cluster = []
    logits_center = []
    for y in tqdm(labels):
        real_images = train_dataset.get_single_class(y.item()).cuda()

        normalized_real_images = train_dataset.normalize(real_images)
        cropped_real_images = crop(normalized_real_images)  

        logits = torch.cat(
            [model(chunk) for chunk in torch.split(cropped_real_images, 100)]
        )

        center = logits.mean(dim=0, keepdim=False)
        logits_center.append(center.squeeze(dim=0))

        cluster = get_kmeans_centers(logits, num_clusters=ipc)
        logits_cluster.append(cluster)

    logits_center = torch.stack(logits_center)

    logits_cluster = torch.stack(logits_cluster)
    logits_cluster = logits_cluster.permute(1, 0, 2).reshape(-1, logits_cluster.size(2))

    # print('logits_center.shape:', logits_center.shape)
    # print('logits_cluster.shape:', logits_cluster.shape)

    return logits_center, logits_cluster


def main(cfg: StatisticCfg):
    save_directory = os.path.join(
        "baselines", "results", "statistic", cfg.dataset, cfg.model
    )
    save_file = os.path.join(save_directory, f"data_ipc{cfg.ipc}.pth")
    if os.path.exists(save_file) and cfg.skip_if_exists:
        print("This eval already done.")
        print("Exiting...")
        exit()

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode="center",
        data_root=cfg.data_root,
    )

    distributed=torch.cuda.device_count() > 1
    eval_model, num_feats = get_model(
        cfg.model, distributed=distributed
    )

    labels = torch.cat(
        [
            torch.tensor([c] * 1, dtype=torch.long)
            for c in range(train_dataset.num_classes)
        ],
        dim=0,
    ).cuda()

    logits_center, logits_cluster = get_cluster_logits(
        labels=labels, model=eval_model, train_dataset=train_dataset, ipc=cfg.ipc
    )

    os.makedirs(save_directory, exist_ok=True)

    save_dict = {
        "statistic_center": logits_center.cpu(),
        "statistic_cluster": logits_cluster.cpu(),
        "labels": labels.cpu(),
    }
    torch.save(save_dict, os.path.join(save_directory, f"data_ipc{cfg.ipc}.pth"))

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg = StatisticCfg(explicit_bool=True).parse_args()
    main(cfg)
