import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

def plot_gaussian_2d(ax, mean, cov, intensity, color='blue', alpha=0.5):
    """
    Plot a 2D Gaussian as an ellipse on a Matplotlib axis.
    :param ax: Matplotlib axis.
    :param mean: Mean of the Gaussian (2D).
    :param cov: Covariance matrix of the Gaussian (2x2).
    :param intensity: Intensity value for color mapping.
    :param color: Base color for the ellipse.
    :param alpha: Transparency level.
    """
    from numpy.linalg import eig

    # Eigen decomposition for the covariance matrix
    eig_vals, eig_vecs = eig(cov)
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eig_vals)  # Scale for 1 standard deviation

    # Modulate color intensity
    rgba_color = plt.cm.viridis(intensity)  # Use a colormap like viridis

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      color=rgba_color, alpha=alpha, edgecolor=color)
    ax.add_patch(ellipse)


def plot_gmm_on_image_2d(gmm, image_path, AABB, output_path, should_show=False):
    img = Image.open(image_path)
    img = np.asarray(img)

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], zorder=0)

    for component in gmm:
        mean = np.array(component['mean'][:2])  # Use only the first two dimensions
        intensity = component['mean'][2]  # Use the third dimension for intensity
        cov = np.array([row[:2] for row in component['covariance'][:2]])  # 2D covariance matrix
        plot_gaussian_2d(ax, mean, cov, intensity=intensity, alpha=0.5)

    plt.savefig(output_path)
    if should_show:
        plt.show()