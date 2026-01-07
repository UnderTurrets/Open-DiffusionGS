export OPENCV_IO_ENABLE_OPENEXR=1
source activate diffusiongs

# 修改为你的实际输出路径
# 这个路径在 eval 完成后会自动生成
exp_root_dir="outputs/diffusion_gs_scene_subset_eval/diffusion-gs-model-scene+lr0.0001@20251224-165126/save/it0"

# chunk 参数控制一次处理多少图像，根据你的GPU内存调整
python eval_scene_result.py --path ${exp_root_dir} --chunk 32

#python eval_scene_result.py --path ${exp_root_dir} --chunk 64