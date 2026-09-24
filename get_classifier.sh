
CUDA_VISIBLE_DEVICES=1 python -m baselines.full_dataset  \
--model dinov2_vitb \
--dataset imagenet-100 \
--data_root /root/datasets  \
--num_eval 1 \
--eval_epochs 10