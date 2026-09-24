CUDA_VISIBLE_DEVICES=0 python -m distillation.eval_syn \
--model dinov2_vitb \
--eval_model eva02_vitb \
--num_eval 1 \
--dataset imagenet-100 \
--data_root /root/datasets  \
--job_tag distillation \
--method sfm \
--eval_mode normal \
--ipc 1


CUDA_VISIBLE_DEVICES=0 python -m distillation.eval_syn \
--model dinov2_vitb \
--eval_model eva02_vitb \
--num_eval 1 \
--dataset imagenet-100 \
--data_root /root/datasets  \
--classifier_path baselines/results/full_dataset/imagenet-100/dinov2_vitb/dinov2_vitb.pth \
--job_tag distillation \
--method sfm \
--eval_mode CI \
--ipc 1

