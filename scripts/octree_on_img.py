from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches


def plot_octree_on_image(image_path, octree, output_path, show_plot=False):
    img = Image.open(image_path)
    img = np.asarray(img)

    nodes = octree["leaf_nodes"]
    AABB = aabb = octree["main_aabb"]

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], zorder=0)

    for aabb in nodes:
        min_point = (aabb["min"][0], aabb["min"][1])
        width = aabb["max"][0] - aabb["min"][0]
        height = aabb["max"][1] - aabb["min"][1]

        rectangle = patches.Rectangle(min_point, width, height, edgecolor='blue', facecolor='none', linewidth=1)
        ax.add_patch(rectangle)

    plt.savefig(output_path)
    if show_plot:
        plt.show()


