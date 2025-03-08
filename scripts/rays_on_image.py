from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

def plot_rays_on_image(image_path, AABB, nodes, intersection_data):
    # we need to have nodes drawn
    # then we need to draw each intersection
    # and then the ray (begining and the whole line + direction)
    img = Image.open(image_path)
    img = np.asarray(img)

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

    for data in intersection_data:
        origin = data["origin"]
        direction = data["direction"]
        intersection = data["intersection"]
        t_values = np.linspace(-10, 10, 100)
        x_values = origin[0] + t_values * direction[0]
        y_values = origin[1] + t_values * direction[1]

        ax.plot(x_values, y_values, 'b-')
        ax.scatter(origin[0], origin[1], color='red')
        ax.scatter(intersection[0], intersection[1], color='yellow')

    plt.show()
