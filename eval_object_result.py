# python eval_object_result.py --path ./outputs/diffusion_gs_obja_4views_relposes_eval/diffusion-gs-model+lr1e-05@20260104-114249/it0

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

def collect_metrics(result_dir: str):
    """
    递归收集指定目录下所有JSON文件中的指标（psnr, ssim, lpips）
    
    Args:
        result_dir: 结果目录路径，包含子目录和metrics.json文件
    
    Returns:
        all_psnr, all_ssim, all_lpips: 分别为包含所有psnr/ssim/lpips值的列表
    """
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    # 递归遍历所有目录
    json_files = list(Path(result_dir).rglob("*_metrics.json"))
    
    if not json_files:
        print(f"警告：在 {result_dir} 下未找到 *_metrics.json 文件")
        return all_psnr, all_ssim, all_lpips
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in tqdm(json_files, desc="处理JSON文件"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # 从JSON中提取三个指标
            if 'psnr' in data and 'ssim' in data and 'lpips' in data:
                all_psnr.append(data['psnr'])
                all_ssim.append(data['ssim'])
                all_lpips.append(data['lpips'])
            else:
                print(f"警告：{json_file} 中缺少必要的字段")
                
        except Exception as e:
            print(f"错误：读取 {json_file} 失败 - {e}")
    
    return all_psnr, all_ssim, all_lpips


def compute_statistics(values):
    """
    计算统计信息
    
    Args:
        values: 数值列表
    
    Returns:
        包含mean, std, min, max的字典
    """
    if not values:
        return None
    
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'count': len(values)
    }


def save_results(output_path: str, psnr_stats, ssim_stats, lpips_stats, 
                 all_psnr, all_ssim, all_lpips):
    """
    保存结果到JSON文件
    
    Args:
        output_path: 输出文件路径
        psnr_stats, ssim_stats, lpips_stats: 统计信息
        all_psnr, all_ssim, all_lpips: 所有原始值
    """
    result_json = {
        'psnr': psnr_stats,
        'ssim': ssim_stats,
        'lpips': lpips_stats,
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=4)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='收集并统计训练评估中的指标')
    
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='包含metrics.json文件的结果目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出结果的JSON文件路径（默认为result_dir/eval_result.json）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，使用默认值
    output_path = args.output if args.output else os.path.join(args.path, 'eval_result.json')
    
    print(f"开始从 {args.path} 收集指标...")
    
    # 收集指标
    all_psnr, all_ssim, all_lpips = collect_metrics(args.path)
    
    if not all_psnr:
        print("未找到任何有效的指标数据")
        return
    
    # 计算统计信息
    psnr_stats = compute_statistics(all_psnr)
    ssim_stats = compute_statistics(all_ssim)
    lpips_stats = compute_statistics(all_lpips)
    
    # 打印结果
    print("\n" + "="*50)
    print("PSNR 统计:")
    print(f"  平均值: {psnr_stats['mean']:.6f}")
    print(f"  标准差: {psnr_stats['std']:.6f}")
    print(f"  最小值: {psnr_stats['min']:.6f}")
    print(f"  最大值: {psnr_stats['max']:.6f}")
    print(f"  样本数: {psnr_stats['count']}")
    
    print("\nSSIM 统计:")
    print(f"  平均值: {ssim_stats['mean']:.6f}")
    print(f"  标准差: {ssim_stats['std']:.6f}")
    print(f"  最小值: {ssim_stats['min']:.6f}")
    print(f"  最大值: {ssim_stats['max']:.6f}")
    print(f"  样本数: {ssim_stats['count']}")
    
    print("\nLPIPS 统计:")
    print(f"  平均值: {lpips_stats['mean']:.6f}")
    print(f"  标准差: {lpips_stats['std']:.6f}")
    print(f"  最小值: {lpips_stats['min']:.6f}")
    print(f"  最大值: {lpips_stats['max']:.6f}")
    print(f"  样本数: {lpips_stats['count']}")
    print("="*50)
    
    # 保存结果
    save_results(output_path, psnr_stats, ssim_stats, lpips_stats, 
                 all_psnr, all_ssim, all_lpips)


if __name__ == "__main__":
    main()
