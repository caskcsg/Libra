import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import os
import shutil
from datasets import load_dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Build image transformation pipeline.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the closest aspect ratio from target ratios.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    """
    Preprocess image with dynamic aspect ratio handling.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    Load and preprocess an image file.
    """
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Calculate frame indices for video sampling.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load and preprocess a video file.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data JSON file")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory containing video files")
    parser.add_argument("--output_root", type=str, required=True, help="Directory to save prediction results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--machine_num", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--machine_rank", type=int, default=1, help="Rank of current GPU (1-based)")
    parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    print(f'Total samples in {args.data_path}: {len(dataset)}')

    dataset = dataset.shuffle(seed=args.seed)
    if args.machine_num > 1:
        dataset = dataset.shard(
            num_shards=args.machine_num, index=args.machine_rank - 1
        )
    print(f"Samples in current process: {len(dataset)}")
    print(f"Sample data:", dataset[0])

    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    model_path = args.model_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    output_path = os.path.join(args.output_root, f"{args.machine_rank}-of-{args.machine_num}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for index, data in tqdm(enumerate(dataset)):
            video_path = os.path.join(args.video_root, data["video"])
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=16, max_num=1
            )
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            video_prefix = "".join(
                [f"Frame-{i+1}: <image>\n" for i in range(len(num_patches_list))]
            )
            if data["subtitle"] == "**********":
                question = (
                    video_prefix
                    + "The above is a video. Please accurately identify the emotional label expressed by the people in the video. Emotional labels include should be limited to: happy, excited, angry, disgusted, hateful, surprised, amazed, frustrated, sad, fearful, despairful, ironic, neutral. The output format should be:\n[label]\n[explanation]"
                )
            else:
                question = (
                    video_prefix
                    + "The above is a video. The video's subtitle is "
                    + f"'{data['subtitle']}'"
                    + ", which maybe the words spoken by the person. Please accurately identify the emotional label expressed by the people in the video. Emotional labels include should be limited to: happy, excited, angry, disgusted, hateful, surprised, amazed, frustrated, sad, fearful, despairful, ironic, neutral. The output format should be:\n[label]\n[explanation]"
                )
            response, history = model.chat(
                tokenizer,
                pixel_values,
                question,
                dict(max_new_tokens=args.max_tokens, do_sample=False),
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            data["answer"] = response
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
            file.flush()

    print("Prediction completed") 

