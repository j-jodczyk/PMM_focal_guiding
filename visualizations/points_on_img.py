from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict
import csv


def plot_points_on_img(image_path, AABB, points):
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
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    ax.scatter(X, Y, c='yellow', alpha=0.1, s=3, zorder=1)

    # Set axis labels and show
    plt.show()

def load_points_from_file(points_file):
    points = []
    with open(points_file) as fh:
        for line in fh:
            i += 1
            if i == 10000:
                break
            point = line.split(' ')
            if (len(point) != 3):
                continue
            points.append((float(point[0]), float(point[1])))
    return points

def plot_points_in_itrations(image_path, iterations, AABB):
    img = Image.open(image_path)
    img = np.asarray(img)

    num_of_iterations = len(iterations)

    fig, axes = plt.subplots(num_of_iterations, 1, figsize=(10, 8 * num_of_iterations))

    if num_of_iterations == 1:
        axes = [axes]  # Ensure axes is iterable if only one plot

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]

    for i, points in enumerate(iterations):
        ax = axes[i]
        ax.imshow(img, extent=[min_x, max_x, min_y, max_y], zorder=0)
        X = [p[0] for p in points]
        Y = [p[1] for p in points]
        ax.scatter(X, Y, c='yellow', alpha=0.1, s=3, zorder=1)
        ax.set_title(f"Iteration {i}")

    plt.tight_layout()
    plt.show()


def load_points_from_iterations(points_file):
    data = defaultdict(list)

    # Read the CSV file
    with open(points_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            iteration = int(row['iterations'])
            x, y, z = float(row['x']), float(row['y']), float(row['z'])
            data[iteration].append((x, y, z))

    # Convert dictionary values to list of lists
    return list(data.values())


def main():
    file_path = "../scenes/dining-room/Reference.png"
    points_path = "./points_in_iterations.csv"
    iterations = load_points_from_iterations(points_path)
    AABB = { "min" : [-6.3658, -1.5274, -4.73046], "max" : [5.27597, 8.04131, 10.0598] }
    plot_points_in_itrations(file_path, iterations[:3], AABB)
    # points = load_points_from_file(points_path)
    # plot_points_on_img(file_path, AABB, points)

if __name__ == "__main__":
    main()
