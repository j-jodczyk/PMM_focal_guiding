import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def plot_gaussians_2d(ax, gaussians, aabb, alpha=0.5):
    x_min, x_max, y_min, y_max = aabb

    # create a grid
    x, y = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]
    pos = np.dstack((x, y))
    # Initialize an empty density grid
    z_total = np.zeros_like(x, dtype=float)

    for g in gaussians:
        rv = multivariate_normal(g["mean"][:2], np.array([row[:2] for row in g['covariance'][:2]]))
        z_total += rv.pdf(pos)  # Sum up PDFs

        mean = g["mean"]
        ax.text(mean[0], mean[1], f'{g["weight"]:.2f}', color='white', fontsize=8,
            ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.contourf(x, y, z_total, levels=30, cmap="viridis", alpha=alpha)

def plot_gaussians_3d(gmm, image_path, AABB, output_path, should_show=False):

    def plot_gaussian_ellipsoid(ax, mean, covariance, color='r', alpha=0.3):
        """ Mimics MATLAB's PLOT_GAUSSIAN_ELLIPSOID function for 3D GMM visualization. """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Generate unit sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=2)

        # Scale by eigenvalues (sqrt because covariance represents squared spread)
        scaling_matrix = np.diag(np.sqrt(eigenvalues))
        transformed_sphere = sphere @ scaling_matrix @ eigenvectors.T

        # Translate to the mean position
        transformed_sphere += mean

        # Extract coordinates
        x_transformed = transformed_sphere[:, :, 0]
        y_transformed = transformed_sphere[:, :, 1]
        z_transformed = transformed_sphere[:, :, 2]

        # Plot the ellipsoid
        ax.plot_surface(x_transformed, y_transformed, z_transformed, color=color, alpha=alpha, edgecolor='k')

    def plot_gmm_ellipsoids(gmm, ax):
        """ Plots ellipsoids for each Gaussian component in the GMM. """
        colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Color for each Gaussian
        for (i, g) in enumerate(gmm):
            plot_gaussian_ellipsoid(ax, g["mean"], g["covariance"], color=colors[i % len(colors)], alpha=0.3)

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    min_z = AABB["min"][2]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]
    max_z = AABB["max"][2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(min_x, max_x)  # X-axis range
    ax.set_ylim(min_y, max_y)  # Y-axis range
    ax.set_zlim(min_z, max_z)  # Z-axis range


    # Plot Gaussian ellipsoids
    plot_gmm_ellipsoids(gmm, ax)

    plt.show()



def plot_gmm_on_image_2d(gmm, image_path, AABB, output_path, should_show=False):
    img = Image.open(image_path)
    img = np.asarray(img)

    min_x = AABB["min"][0]
    min_y = AABB["min"][1]
    max_x = AABB["max"][0]
    max_y = AABB["max"][1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, extent=[min_x, max_x, min_y, max_y], zorder=0)
    ax.axis('off')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plot_gaussians_2d(ax, gmm, [min_x, max_x, min_y, max_y], alpha=0.5)
    # plt.show()

    plt.savefig(output_path, bbox_inches='tight')