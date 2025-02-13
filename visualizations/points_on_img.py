from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def plot_points_on_img(image_path, AABB, points_file):
    # Load image
    img = Image.open(image_path)
    img = np.asarray(img)

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], zorder=0)  # Adjust extents as needed

    # Plot each GMM component
    X = []
    Y = []
    i = 0
    with open(points_file) as fh:
        for line in fh:
            i += 1
            if i == 10000:
                break
            point = line.split(' ')
            if (len(point) != 3):
                continue
            X.append(float(point[0]))
            Y.append(float(point[1]))
    ax.scatter(X, Y, c='yellow', alpha=0.1, s=3, zorder=1)

    # Set axis labels and show
    plt.show()

def main():
    file_path = "../scenes/dining-room/Reference.png"
    points_path = "./samples_thread_7.txt"
    AABB = { "min" : [-6.3658, -1.5274, -4.73046], "max" : [5.27597, 8.04131, 10.0598] }
    plot_points_on_img(file_path, AABB, points_path)

if __name__ == "__main__":
    main()
