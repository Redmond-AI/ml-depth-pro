#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # Save 32-bit float TIFF file without normalization.
        if args.output_path is not None:
            output_file = (
                args.output_path
                / image_path.relative_to(relative_path).parent
                / (image_path.stem + ".tiff")
            )
            LOGGER.info(f"Saving unnormalized depth to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            depth_image = PIL.Image.fromarray(depth.astype(np.float32), mode='F')
            depth_image.save(output_file, format="TIFF")
    
        # Return the output TIFF file path
        return output_file

    output_tiff_files = []

    # Use ThreadPoolExecutor to process images concurrently.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_image, image_path) for image_path in image_paths]
        for future in tqdm(as_completed(futures), total=len(image_paths)):
            output_file = future.result()
            if output_file:
                output_tiff_files.append(output_file)

    LOGGER.info("Processing complete. Computing global min and max.")

    # Use the collected list of output TIFF files
    # Step 2: Find global min and max from all TIFF files.
    global_min = float('inf')
    global_max = float('-inf')
    for depth_file in output_tiff_files:
        depth = np.array(PIL.Image.open(depth_file), dtype=np.float32)
        current_min = depth.min()
        current_max = depth.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    LOGGER.info(f"Global min depth: {global_min}, Global max depth: {global_max}")

    # Step 3 and 4: Normalize depth maps, save as 16-bit PNG, and delete TIFF files.
    for depth_file in output_tiff_files:
        depth = np.array(PIL.Image.open(depth_file), dtype=np.float32)
        normalized_depth = (depth - global_min) / (200 - global_min)
        depth_16bit = (normalized_depth * 65535).astype(np.uint16)  # Change to 16-bit
        output_file = depth_file.with_suffix('.png')
        LOGGER.info(f"Saving normalized 16-bit depth to: {str(output_file)}")
        PIL.Image.fromarray(depth_16bit, mode='I;16').save(output_file, format="PNG")  # Change mode to 'I;16'
        LOGGER.info(f"Deleting temporary TIFF file: {str(depth_file)}")
        depth_file.unlink()

    LOGGER.info("All depth images have been normalized and saved as 16-bit PNGs. TIFF files have been deleted.")

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
