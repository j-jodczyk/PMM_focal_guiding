import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

def extract_parameters(folder_name):
    """Extracts numerical parameter values from the folder name."""
    pattern = r"gmmSplittingThreshold_([0-9\.]+)_gmmMergingThreshold_([0-9\.]+)"
    match = re.search(pattern, folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

def get_latest_date_folder(base_path):
    """Finds the most recent date folder in the visualizations directory."""
    vis_path = os.path.join(base_path, "visualizations")
    if not os.path.exists(vis_path):
        return None
    date_folders = [d for d in os.listdir(vis_path) if re.match(r'\d{2}-\d{2}', d)]
    if not date_folders:
        return None
    latest_date = max(date_folders, key=lambda d: datetime.strptime(d, "%d-%m"))
    return os.path.join(vis_path, latest_date, "gmm")

def get_latest_iteration_image(folder_path):
    """Finds the latest iteration image in the given folder."""
    if not os.path.exists(folder_path):
        return None
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return None
    latest_image = max(image_files, key=lambda f: int(os.path.splitext(f)[0]))
    return Image.open(os.path.join(folder_path, latest_image))

def load_images(base_path):
    """Loads images and organizes them by parameter values."""
    image_data = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            params = extract_parameters(folder)
            if params:
                latest_gmm_path = get_latest_date_folder(folder_path)
                if latest_gmm_path:
                    latest_image = get_latest_iteration_image(latest_gmm_path)
                    if latest_image is not None:
                        image_data.append((params, latest_image))
    return image_data

def plot_image_matrix(image_data):
    """Plots images in a matrix sorted by parameters."""
    if not image_data:
        print("No valid images found.")
        return

    unique_splitting = sorted(set(p[0] for p, _ in image_data))
    unique_merging = sorted(set(p[1] for p, _ in image_data))

    fig = plt.figure(figsize=(len(unique_merging), len(unique_splitting)))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(len(unique_splitting), len(unique_merging)),
        axes_pad=0.1
        )

    for ax, ((split, merge), img) in zip(grid, image_data):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    # Set axis labels
    # for i, split in enumerate(unique_splitting):
    #     fig.text(0.02, 0.85 - i * (1 / len(unique_splitting)), f"S: {split}", fontsize=10, va='center')
    # for j, merge in enumerate(unique_merging):
    #     fig.text(0.15 + j * (0.8 / len(unique_merging)), 0.98, f"M: {merge}", fontsize=10, ha='center')

    plt.show()


# Usage example:
base_path = "../argument_tuning/dining-room"  # Change this to your actual path
image_data = load_images(base_path)
plot_image_matrix(image_data)
