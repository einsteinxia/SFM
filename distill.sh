CUDA_VISIBLE_DEVICES=0 python -m distillation.distill \
--model dinov2_vitb \
--dataset imagenet-100 \
--data_root /root/datasets \
--ipc 1 \
--job_tag distillation \
--augs_per_batch 1  \
--statistic_path baselines/results/statistic/imagenet-100/dinov2_vitb/data_ipc1.pth \
--method sfm 

