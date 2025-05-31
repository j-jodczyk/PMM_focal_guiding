import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from scipy.stats import multivariate_normal
import sys
import re
from log_parser import parse_log_file
from PIL import Image


def parse_defaults(root):
    """Parse <default> elements and return a dict of {name: value}."""
    defaults = {}
    for d in root.findall("default"):
        name = d.attrib["name"]
        value = d.attrib["value"]
        # Try to convert to int or float, fallback to string
        if re.match(r"^\d+$", value):
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        defaults[name] = value
    return defaults

def resolve_value(value, defaults):
    """Replace $param in value with actual defaults[param]."""
    if isinstance(value, str) and value.startswith("$"):
        param = value[1:]
        if param in defaults:
            return defaults[param]
        else:
            raise ValueError(f"Parameter {param} not found in defaults")
    return value

def parse_sensor_info(scene_path):
    tree = ET.parse(scene_path)
    root = tree.getroot()

    defaults = parse_defaults(root)

    sensor = root.find(".//sensor")
    film = sensor.find("film")
    
    width = resolve_value(film.find("integer[@name='width']").attrib["value"], defaults)
    height = resolve_value(film.find("integer[@name='height']").attrib["value"], defaults)
    
    # Convert to int if still string
    width = int(width) if isinstance(width, str) else width
    height = int(height) if isinstance(height, str) else height

    aspect = width / height

    fov_x_deg = float(sensor.find("float[@name='fov']").attrib["value"])
    fov_x_rad = np.deg2rad(fov_x_deg)
    tan_fov_x_2 = np.tan(fov_x_rad / 2)
    tan_fov_y_2 = tan_fov_x_2 / aspect

    transform = sensor.find("transform[@name='toWorld']")
    to_world = build_transform_matrix(transform)
    to_camera = np.linalg.inv(to_world)

    return {
        "width": width,
        "height": height,
        "tan_fov_x_2": tan_fov_x_2,
        "tan_fov_y_2": tan_fov_y_2,
        "to_camera": to_camera
    }


def build_transform_matrix(transform_element):
    def rotation_matrix(axis, angle_deg):
        angle = np.deg2rad(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            return np.array([
                [1, 0,  0, 0],
                [0, c, -s, 0],
                [0, s,  c, 0],
                [0, 0,  0, 1]
            ])
        elif axis == 'y':
            return np.array([
                [ c, 0, s, 0],
                [ 0, 1, 0, 0],
                [-s, 0, c, 0],
                [ 0, 0, 0, 1]
            ])
        elif axis == 'z':
            return np.array([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
            ])

    matrix = np.identity(4)
    for child in transform_element:
        if child.tag == 'translate':
            x = float(child.attrib.get('x', 0))
            y = float(child.attrib.get('y', 0))
            z = float(child.attrib.get('z', 0))
            T = np.identity(4)
            T[:3, 3] = [x, y, z]
            matrix = T @ matrix
        elif child.tag == 'rotate':
            angle = float(child.attrib['angle'])
            axis = 'x' if 'x' in child.attrib else 'y' if 'y' in child.attrib else 'z'
            R = rotation_matrix(axis, angle)
            matrix = R @ matrix
    return matrix


def project_points(points_3d, cam_info):
    """
    points_3d: (N, 3) array of world coordinates
    cam_info: output of parse_sensor_info()
    Returns: (N, 2) array of pixel coordinates
    """
    N = points_3d.shape[0]
    homo_points = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    cam_points = (cam_info["to_camera"] @ homo_points.T).T  # (N, 4)

    x_cam = cam_points[:, 0]
    y_cam = cam_points[:, 1]
    z_cam = cam_points[:, 2]

    # Perspective divide
    x_ndc = (x_cam / -z_cam) / cam_info["tan_fov_x_2"]
    y_ndc = (y_cam / -z_cam) / cam_info["tan_fov_y_2"]

    # Convert to pixel coordinates
    px = (x_ndc + 1) * 0.5 * cam_info["width"]
    py = (1 - (y_ndc + 1) * 0.5) * cam_info["height"]

    return np.stack([px, py], axis=1)

def compute_jacobian(cam_point, cam_info):
    """
    cam_point: (3,) in camera space (x_c, y_c, z_c)
    cam_info: dict with tan_fov_x_2, tan_fov_y_2, width, height
    
    Returns:
        J: (2,3) Jacobian matrix from camera 3D coords to pixel 2D coords
    """
    x_c, y_c, z_c = cam_point
    w, h = cam_info["width"], cam_info["height"]
    tan_fx, tan_fy = cam_info["tan_fov_x_2"], cam_info["tan_fov_y_2"]

    # Partial derivatives for u and v
    dpx_dx = (w / 2) / tan_fx / (-z_c)
    dpx_dy = 0
    dpx_dz = (w / 2) * (x_c / (z_c**2)) / tan_fx

    dpy_dx = 0
    dpy_dy = -(h / 2) / tan_fy / (-z_c)
    dpy_dz = (h / 2) * (y_c / (z_c**2)) / tan_fy

    # Note the sign for dpy_dz due to y-flip
    J = np.array([
        [dpx_dx, dpx_dy, dpx_dz],
        [dpy_dx, dpy_dy, dpy_dz]
    ])
    return J


def project_gmm(means, covariances, cam_info):
    """
    means: (N,3) world-space means
    covariances: (N,3,3) world-space covariances
    cam_info: parsed camera info dict
    
    Returns:
        pixel_means: (N,2) pixel coordinates
        pixel_covs: (N,2,2) pixel covariance matrices for ellipse plotting
    """
    N = means.shape[0]
    homo_points = np.hstack([means, np.ones((N, 1))])
    cam_points = (cam_info["to_camera"] @ homo_points.T).T
    cam_means = cam_points[:, :3]

    # Extract rotation (top-left 3x3) of to_camera (world->camera)
    R = cam_info["to_camera"][:3, :3]

    pixel_means = []
    pixel_covs = []

    for i in range(N):
        mu_c = cam_means[i]
        Sigma_w = covariances[i]

        # Transform covariance to camera space
        Sigma_c = R @ Sigma_w @ R.T

        # Compute Jacobian of projection at mean
        J = compute_jacobian(mu_c, cam_info)

        # Project covariance to 2D pixel space
        Sigma_2d = J @ Sigma_c @ J.T

        # Project mean to pixel space (reuse earlier logic)
        x_ndc = (mu_c[0] / -mu_c[2]) / cam_info["tan_fov_x_2"]
        y_ndc = (mu_c[1] / -mu_c[2]) / cam_info["tan_fov_y_2"]
        px = (x_ndc + 1) * 0.5 * cam_info["width"]
        py = (1 - (y_ndc + 1) * 0.5) * cam_info["height"]

        pixel_means.append([px, py])
        pixel_covs.append(Sigma_2d)

    return np.array(pixel_means), np.array(pixel_covs)

def create_gmm_density_map(pixel_means, pixel_covs, weights, width, height, sigma_scale=3):
    """
    Create a 2D density map of the GMM on image pixel grid.

    sigma_scale: number of std deviations to consider for kernel support
    """
    Y, X = np.mgrid[0:height, 0:width]
    pos = np.dstack((X, Y))

    density = np.zeros((height, width), dtype=np.float32)

    for mu, cov, w in zip(pixel_means, pixel_covs, weights):
        # Scale covariance by sigma_scale^2 for the kernel size
        cov_scaled = cov * (sigma_scale ** 2)

        # Create a multivariate normal with mean=mu, cov=cov_scaled
        rv = multivariate_normal(mean=mu, cov=cov_scaled)

        # Evaluate density on the grid
        density += w * rv.pdf(pos)

    return density

def plot_gmm_heatmap_overlay(image_path, pixel_means, pixel_covs, weights, alpha=0.5):
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    density = create_gmm_density_map(pixel_means, pixel_covs, weights, w, h)

    # Normalize density for coloring
    norm = Normalize(vmin=0, vmax=np.max(density))
    density_norm = norm(density)

    # Apply colormap
    cmap = plt.cm.inferno  # or viridis, plasma, magma...
    heatmap = cmap(density_norm)

    # Overlay heatmap on image with alpha blending
    overlay = heatmap[..., :3] * 255 * alpha + img_np * (1 - alpha)
    overlay = overlay.astype(np.uint8)

    plt.figure(figsize=(w/100, h/100), dpi=100)
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



# Example usage:
if __name__ == "__main__":
    scene_path = sys.argv[3]
    img_path = sys.argv[2]
    log_file = sys.argv[1]
    cam_info = parse_sensor_info(scene_path)

    GMMs, *_ = parse_log_file(log_file)
    for GMM in GMMs:
        means3D = np.array([np.array(comp['mean']) for comp in GMM])
        covs3D = np.array([np.array(comp['covariance']) for comp in GMM])
        weights = np.array([comp['weight'] for comp in GMM])

        means, covs = project_gmm(means3D, covs3D, cam_info)
        plot_gmm_heatmap_overlay(img_path, means, covs, weights)
