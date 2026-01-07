#!/usr/bin/env python3
"""
为子集数据生成 evaluation_index_re10k.json
为每个场景随机选择固定的输入视角和目标视角
python dataset_download/real10k_generate_eval_index.py \
    --metadata_dir /root/autodl-tmp/re10k_subset_diffusionGS/test/metadata \
    --output_file dataset_download/evaluation_index_subset.json
"""

import json
import os
import random
import argparse
from pathlib import Path

def generate_evaluation_index(metadata_dir, output_file, num_context=1, num_target=3, seed=42):
    """
    生成评估索引文件
    
    Args:
        metadata_dir: metadata 目录路径
        output_file: 输出 JSON 文件路径
        num_context: context 视角数量（通常为1，作为输入）
        num_target: target 视角数量（作为渲染目标）
        seed: 随机种子，保证可重复性
    """
    random.seed(seed)
    
    metadata_dir = Path(metadata_dir)
    evaluation_index = {}
    
    # 遍历所有 JSON 元数据文件
    json_files = sorted(metadata_dir.glob("*.json"))
    print(f"Found {len(json_files)} scenes in {metadata_dir}")
    
    for json_file in json_files:
        scene_name = json_file.stem  # 文件名不含扩展名
        
        # 读取场景元数据
        with open(json_file, 'r') as f:
            scene_data = json.load(f)
        
        num_frames = len(scene_data['frames'])
        
        # 需要至少 num_context + num_target 个视角
        required_views = num_context + num_target
        if num_frames < required_views:
            print(f"Warning: Scene {scene_name} has only {num_frames} frames, "
                  f"need at least {required_views}. Skipping...")
            continue
        
        # 随机选择视角索引
        all_indices = list(range(num_frames))
        selected_indices = random.sample(all_indices, required_views)
        
        # 分配为 context 和 target
        context_indices = selected_indices[:num_context]
        target_indices = selected_indices[num_context:]
        
        evaluation_index[scene_name] = {
            "context": context_indices,
            "target": target_indices
        }
        
        print(f"Scene {scene_name}: {num_frames} frames -> "
              f"context {context_indices}, target {target_indices}")
    
    # 保存到 JSON 文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(evaluation_index, f, indent=2)
    
    print(f"\nGenerated evaluation index for {len(evaluation_index)} scenes")
    print(f"Saved to: {output_file}")
    
    return evaluation_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation index for RealEstate10K subset")
    parser.add_argument("--metadata_dir", type=str, required=True,
                       help="Path to metadata directory")
    parser.add_argument("--output_file", type=str, default="evaluation_index_subset.json",
                       help="Output JSON file path")
    parser.add_argument("--num_context", type=int, default=1,
                       help="Number of context views (input)")
    parser.add_argument("--num_target", type=int, default=3,
                       help="Number of target views (render targets)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_evaluation_index(
        metadata_dir=args.metadata_dir,
        output_file=args.output_file,
        num_context=args.num_context,
        num_target=args.num_target,
        seed=args.seed
    )
