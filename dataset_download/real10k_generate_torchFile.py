'''
需要准备的原始 RealEstate10K 目录格式
/path/to/Real-Estate-10k/
├── train/
│   ├── <scene_id>/              # 场景文件夹，名称是 scene_id
│   │   ├── <timestamp0>.jpg     # 文件名是时间戳（整数），后缀 .jpg
│   │   ├── <timestamp1>.jpg
│   │   └── ...
│   └── ...
├── test/
│   └── <scene_id>/...
└── metadata
    ├── train/
    │   ├── <scene_id>.txt       # 元数据文件，文件名与 scene_id 对应
    │   └── ...
    └── test/
        └── <scene_id>.txt
        
在脚本开头修改为你的实际路径（或用软链接指向相同结构）：
INPUT_IMAGE_DIR = Path("/data/scene-rep/Real-Estate-10k")                # 原始图像根目录
INPUT_METADATA_DIR = Path("/data/scene-rep/Real-Estate-10k/metadata")  # 元数据根目录
OUTPUT_DIR = Path("/data/scene-rep/Real-Estate-10k/re10k_pt")           # 输出 .torch 目录
TARGET_BYTES_PER_CHUNK = int(1e8)  # 每个 chunk 目标大小，默认 ~100MB

python dataset_download/real10k_generate_torchFile.py
'''
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

INPUT_IMAGE_DIR = Path("/data/scene-rep/Real-Estate-10k")
INPUT_METADATA_DIR = Path("/data/scene-rep/Real-Estate-10k/metadata")
OUTPUT_DIR = Path("/data/scene-rep/Real-Estate-10k/re10k_pt")

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm((INPUT_IMAGE_DIR / stage).iterdir(), desc="Indexing images")
    )
    metadata_keys = set(
        example.stem
        for example in tqdm(
            (INPUT_METADATA_DIR / stage).iterdir(), desc="Indexing metadata"
        )
    )

    missing_image_keys = metadata_keys - image_keys
    if len(missing_image_keys) > 0:
        print(
            f"Found metadata but no images for {len(missing_image_keys)} examples.",
            file=sys.stderr,
        )
    missing_metadata_keys = image_keys - metadata_keys
    if len(missing_metadata_keys) > 0:
        print(
            f"Found images but no metadata for {len(missing_metadata_keys)} examples.",
            file=sys.stderr,
        )

    keys = image_keys & metadata_keys
    print(f"Found {len(keys)} keys.")
    return keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {int(path.stem): load_raw(path) for path in example_path.iterdir()}


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    with example_path.open("r") as f:
        lines = f.read().splitlines()

    url = lines[0]

    timestamps = []
    cameras = []

    for line in lines[1:]:
        timestamp, *camera = line.split(" ")
        timestamps.append(int(timestamp))
        cameras.append(np.fromstring(",".join(camera), sep=","))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    for stage in ("train", "test"):
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / stage / key
            metadata_dir = INPUT_METADATA_DIR / stage / f"{key}.txt"
            num_bytes = get_size(image_dir)

            # Read images and metadata.
            images = load_images(image_dir)
            example = load_metadata(metadata_dir)

            # Merge the images into the example.
            example["images"] = [
                images[timestamp.item()] for timestamp in example["timestamps"]
            ]
            assert len(images) == len(example["timestamps"])

            # Add the key to the example.
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()