#!/usr/bin/env python3
"""
下载G-Objaverse数据集的子集
用法：python download_gobjaverse_280k_subset.py --num_objects 1000 --category Daily-Used
可选类别：
Daily-Used（日用品）
Buildings & Outdoor
Human-Shape（人形）
Animals（动物）
Transportations
Furnitures（家具）
Electronics
Plants（植物）
Food（食物）
Poor-quality
"""

import json
import argparse
import subprocess
import os

def download_subset(num_objects=1000, category="Daily-Used", save_dir="./gobjaverse_data"):
    """
    下载指定数量的G-Objaverse物体
    
    参数:
        num_objects: 要下载的物体数量
        category: 类别名称 (Human-Shape, Daily-Used, Animals等)
        save_dir: 保存目录
    """

    # 支持的类别
    categories = [
        "Human-Shape", "Animals", "Daily-Used", "Furnitures",
        "Buildings-Outdoor", "Transportations", "Plants", 
        "Food", "Electronics"
    ]
    
    if category not in categories and category != "all":
        print(f"❌ 不支持的类别: {category}")
        print(f"支持的类别: {', '.join(categories)} 或 'all'")
        return
    
    # 下载索引文件
    if category == "all":
        index_url = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k.json"
        index_file = "gobjaverse_280k.json"
    else:
        index_url = f"https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k_split/gobjaverse_280k_{category}.json"
        index_file = f"gobjaverse_280k_{category}.json"
    
    print(f"📥 下载索引文件: {index_url}")
    subprocess.run(["wget", "-nc", index_url], check=True)
    
    # 读取并截取
    print(f"📝 读取索引文件并截取前 {num_objects} 个物体...")
    with open(index_file, 'r') as f:
        data = json.load(f)
    
    subset = data[:num_objects]
    subset_file = f"gobjaverse_subset_{num_objects}.json"
    
    with open(subset_file, 'w') as f:
        json.dump(subset, f)
    
    print(f"✅ 创建子集文件: {subset_file} (包含 {len(subset)} 个物体)")
    
    # 估算空间
    estimated_size_gb = len(subset) * 36 / 1024
    print(f"💾 预计需要空间: ~{estimated_size_gb:.1f} GB")
    
    # 用户确认
    print("\n" + "="*60)
    print("⚠️  请确认以下信息:")
    print(f"  - 物体数量: {len(subset)}")
    print(f"  - 类别: {category}")
    print(f"  - 保存目录: {os.path.abspath(save_dir)}")
    print(f"  - 磁盘空间: ~{estimated_size_gb:.1f} GB")
    print("="*60)
    
    user_confirm = input("\n✅ 确认开始下载？(输入 yes 继续, 其他任何输入取消): ").strip().lower()
    if user_confirm != 'yes':
        print("\n❌ 下载已取消")
        return
    
    print("\n")
    
    # 创建训练/验证/测试集划分
    print(f"📊 创建 train/val/test 划分...")
    os.makedirs("json_files", exist_ok=True)
    
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio = 0.1
    
    train_end = int(len(subset) * train_ratio)
    val_end = int(len(subset) * (train_ratio + val_ratio))
    
    # 提取 uid
    if isinstance(subset[0], dict):
        train_uids = [item.get('uid', item.get('id', str(i))) for i, item in enumerate(subset[:train_end])]
        val_uids = [item.get('uid', item.get('id', str(i))) for i, item in enumerate(subset[train_end:val_end])]
        test_uids = [item.get('uid', item.get('id', str(i))) for i, item in enumerate(subset[val_end:])]
    else:
        train_uids = subset[:train_end]
        val_uids = subset[train_end:val_end]
        test_uids = subset[val_end:]

    full_uids = train_uids + val_uids + test_uids
    
    with open("json_files/train.json", 'w') as f:
        json.dump(train_uids, f)
    with open("json_files/val.json", 'w') as f:
        json.dump(val_uids, f)
    with open("json_files/test.json", 'w') as f:
        json.dump(test_uids, f)
    with open("json_files/full.json", 'w') as f:
        json.dump(full_uids, f)
    
    print(f"  - train.json: {len(train_uids)} 个物体")
    print(f"  - val.json: {len(val_uids)} 个物体")
    print(f"  - test.json: {len(test_uids)} 个物体")
    print(f"  - full.json: {len(full_uids)} 个物体")
    
    # 下载数据
    print(f"\n🚀 开始下载数据到 {save_dir}...")
    print(f"提示: 这可能需要一些时间，请耐心等待...\n")
    
    # 检查是否存在下载脚本
    if not os.path.exists("download_gobjaverse_280k.py"):
        print("⚠️  未找到 download_gobjaverse_280k.py")
        print("请从以下链接下载:")
        print("wget https://raw.githubusercontent.com/modelscope/richdreamer/main/dataset/gobjaverse/download_gobjaverse_280k.py")
        return
    
    download_cmd = [
        "python", "download_gobjaverse_280k.py",
        save_dir,
        subset_file,
        "16"  # 线程
    ]
    
    print(f"📋 执行命令: {' '.join(download_cmd)}")
    print("="*60)
    
    # 直接运行子进程,让其输出原样显示(支持进度条)
    try:
        result = subprocess.run(download_cmd)
        if result.returncode != 0:
            print(f"\n❌ 下载失败,退出码: {result.returncode}")
            return
    except KeyboardInterrupt:
        print("\n⚠️  下载被用户中断")
        return
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        return
    print("="*60)
    
    print("\n✅ 下载完成！")
    print(f"\n📁 数据目录: {save_dir}")
    print(f"📁 JSON文件: json_files/")
    print("\n下一步:")
    print("1. 修改 diffusionGS/configs/diffusionGS_rel.yaml:")
    print(f"   local_dir: '{os.path.abspath('json_files')}'")
    print(f"   image_dir: '{os.path.abspath(save_dir)}/'")
    print("2. 运行训练: bash scripts/train_obj_stage1.sh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载G-Objaverse子集")
    parser.add_argument("--num_objects", type=int, default=1000, 
                        help="要下载的物体数量 (默认: 1000)")
    parser.add_argument("--category", type=str, default="Daily-Used",
                        help="类别名称 (默认: Daily-Used)")
    parser.add_argument("--save_dir", type=str, default="./gobjaverse_data",
                        help="保存目录 (默认: ./gobjaverse_data)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("G-Objaverse 子集下载工具")
    print("=" * 60)
    print(f"物体数量: {args.num_objects}")
    print(f"类别: {args.category}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 60)
    
    download_subset(args.num_objects, args.category, args.save_dir)
