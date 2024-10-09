import argparse
import os
import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio  # Updated import to use imageio v2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Processing Script for Black and White Level Adjustment')
    parser.add_argument('--input', required=True, help='Path to the input folder containing images')
    parser.add_argument('--output', required=True, help='Path to the output folder for processed images')
    parser.add_argument('--black', type=float, default=0.1, help='Float value (0-1) for the black level threshold')
    parser.add_argument('--white', type=float, default=0.9, help='Float value (0-1) for the white level threshold')
    parser.add_argument('--invert', action='store_true', help='Invert image values after normalization')
    return parser.parse_args()

def load_image_paths(input_folder):
    # Supported image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp', '*.gif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    return sorted(image_paths)

def process_image(image_path, device, black_ratio, white_ratio, invert):
    # Open image and convert to tensor
    image = Image.open(image_path)
    image_array = np.array(image)

    # Determine bit depth
    max_value = np.iinfo(image_array.dtype).max
    # Convert image array to float32 before creating tensor
    image_tensor = torch.from_numpy(image_array.astype(np.float32)).to(device, dtype=torch.float32)

    # Normalize image to 0-1 range
    image_tensor /= max_value

    if invert:
        image_tensor = 1.0 - image_tensor

    # Flatten the image for histogram calculation
    flat = image_tensor.view(-1).cpu().numpy()

    # Compute histogram and CDF
    hist_bins = 1000
    hist, bin_edges = np.histogram(flat, bins=hist_bins, range=(0.0, 1.0))
    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]

    # Find black and white thresholds
    black_threshold = bin_edges[np.searchsorted(cdf_normalized, black_ratio)]
    white_threshold = bin_edges[np.searchsorted(cdf_normalized, white_ratio)]

    # Adjust pixels
    adjusted = torch.clamp((image_tensor - black_threshold) / (white_threshold - black_threshold), 0.0, 1.0)

    # Rescale to 16-bit and convert to CPU
    adjusted = (adjusted * 65535).cpu().numpy().astype(np.uint16)

    # Create image from numpy array with appropriate mode
    if image_array.ndim == 2:
        mode = 'I;16'
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        mode = 'RGB'
    else:
        raise ValueError('Unsupported image format')
    adjusted_image = Image.fromarray(adjusted, mode=mode)

    return adjusted_image

def main():
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Load image paths
    image_paths = load_image_paths(args.input)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_images = []
    for img_path in tqdm(image_paths, desc='Processing images'):
        adjusted_image = process_image(img_path, device, args.black, args.white, args.invert)
        # Save image
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(args.output, f"{name}.png")
        adjusted_image.save(output_path, format='PNG')
        processed_images.append(output_path)

    # Create video from processed images
    video_path = os.path.join(args.output, 'output_video.mp4')
    with imageio.get_writer(video_path, fps=30, codec='libx264', quality=10) as writer:
        for img_path in tqdm(processed_images, desc='Creating video'):
            image = imageio.imread(img_path)
            writer.append_data(image)

if __name__ == '__main__':
    main()