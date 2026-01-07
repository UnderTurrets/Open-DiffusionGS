export OPENCV_IO_ENABLE_OPENEXR=1
source activate diffusiongs

# 根据你的GPU数量调整，如果只有1个GPU，使用 --nproc-per-node=1
# 如果有多个GPU，可以使用更多
# TORCHELASTIC_TIMEOUT=18000 torchrun  --standalone --nnodes=1 --nproc-per-node=8 \
#     launch.py --validate --use_ema --gpu 0,1,2,3,4,5,6,7 \
#     --config diffusionGS/configs/diffusionGS_scene_eval.yaml

TORCHELASTIC_TIMEOUT=18000 torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    launch.py --validate --use_ema --gpu 0 \
    --config diffusionGS/configs/diffusionGS_scene_eval.yaml