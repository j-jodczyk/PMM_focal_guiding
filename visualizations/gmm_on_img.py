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

    # Draw the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      color=rgba_color, alpha=alpha, edgecolor=color)
    ax.add_patch(ellipse)

def plot_gmm_on_image_2d(gmm, image_path, AABB):
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
    for component in gmm:
        weight = component['weight']
        mean = np.array(component['mean'][:2])  # Use only the first two dimensions
        intensity = component['mean'][2]  # Use the third dimension for intensity
        cov = np.array([row[:2] for row in component['covaraince'][:2]])  # 2D covariance matrix
        plot_gaussian_2d(ax, mean, cov, intensity=intensity, alpha=0.5)

    # Set axis labels and show
    plt.show()


def main():
    # take image
    file_path = "../scenes/dining-room/Reference.png"
    # take gmm
    GMM = [
        { "weight": 0.742715, "mean": [-0.570314, 2.95218, 2.82582], "covaraince": [[9.96883, -1.82459, -2.20546],[-1.82459, 6.99116, 1.26026],[-2.20546, 1.26026, 14.9656]] },
        { "weight": 0.0110646, "mean": [-0.525618, 3.02775, 2.83586], "covaraince": [[10.0537, -1.43303, -1.81688],[-1.43303, 7.07895, 1.05266],[-1.81688, 1.05266, 15.0719]] },
        { "weight": 0.246221, "mean": [-0.481622, 2.90999, 2.88917], "covaraince": [[9.9898, -1.55807, -2.24916],[-1.55807, 6.9976, 1.70858],[-2.24916, 1.70858, 14.7592]] },
    ]
    AABB = { "min" : [-6.3658, -1.5274, -4.73046], "max" : [5.27597, 8.04131, 10.0598] }
    # appy gmm on image
    plot_gmm_on_image_2d(GMM, file_path, AABB)

if __name__=="__main__":
    main()
