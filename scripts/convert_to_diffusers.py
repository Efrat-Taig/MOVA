import os
import torch
import argparse
from mmengine.config import Config

from mova.registry import DIFFUSION_PIPELINES


def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to diffusers format")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/mova.py",
        help="Config file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save diffusers format model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading model from config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}, Dtype: {args.torch_dtype}")

    # Build diffusion pipeline
    pipe = DIFFUSION_PIPELINES.build(
        cfg.diffusion_pipeline,
        default_args={"device": args.device, "torch_dtype": torch_dtype},
    )
    pipe.audio_vae.remove_weight_norm()

    print("Saving model to diffusers format...")
    os.makedirs(args.output_dir, exist_ok=True)
    pipe.save_pretrained(args.output_dir)

    print(f"Model saved successfully to {args.output_dir}")


if __name__ == "__main__":
    main()
