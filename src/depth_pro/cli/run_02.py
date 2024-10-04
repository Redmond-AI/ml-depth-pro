#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def run(args):
    """Run Depth Pro on a directory of images."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    if not args.image_path.is_dir():
        raise ValueError(f"Input path {args.image_path} is not a directory.")

    image_paths = list(args.image_path.glob("**/*"))
    image_paths = [
        p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    relative_path = args.image_path

    def process_image(image_path):
        """Process a single image."""
        import time

        start_time = time.time()
        try:
            LOGGER.info(f"Loading image {image_path} ...")
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            LOGGER.error(str(e))
            return

        # Run prediction.
        prediction = model.infer(transform(image), f_px=f_px)

        # Extract the depth.
        depth = prediction["depth"].detach().cpu().numpy().squeeze()

        # Save normalized grayscale PNG image.
        if args.output_path is not None:
            output_file = (
                args.output_path
                / image_path.relative_to(relative_path).parent
                / (image_path.stem + ".png")
            )
            LOGGER.info(f"Saving normalized grayscale depth to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
            gray_depth = (normalized_depth * 255).astype(np.uint8)
            PIL.Image.fromarray(gray_depth).save(
                output_file, format="PNG"
            )

        # Print inference time.
        inference_time = time.time() - start_time
        print(f"Inference for {image_path} took {inference_time:.2f} seconds.")

    # Use ThreadPoolExecutor to process images concurrently.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(process_image, image_paths), total=len(image_paths)))

    LOGGER.info("Done predicting depth!")
    if not args.skip_display:
        plt.show(block=True)


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference script of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=Path,
        required=True,  # Changed to make input directory required
        help="Path to input image directory.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,  # Changed to make output directory required
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of worker threads for inference (default: 2).",
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()
