from log_parser import parse_log_file
from gmm_on_img import plot_gmm_on_image_2d
from points_on_img import plot_points_in_itration
from rays_on_image import plot_rays_on_image
import os
from datetime import datetime
import argparse

def create_visualization_folders(root):
    date_str = datetime.today().strftime("%d-%m")
    gmm_dir_path = f"{root}/visualizations/{date_str}/gmm"
    points_dir_path = f"{root}/visualizations/{date_str}/points"
    os.makedirs(gmm_dir_path, exist_ok=True)
    os.makedirs(points_dir_path, exist_ok=True)

    return (gmm_dir_path, points_dir_path)


def main(root_dir):
    log_file_path = f"{root_dir}/mitsuba.DESKTOP-06NEMHS.log"
    image_path = f"{root_dir}/Reference.png"
    GMMs, valid_samples, intersection_data, octrees = parse_log_file(log_file_path)
    aabb = octrees[0]["main_aabb"]
    nodes = octrees[0]["leaf_nodes"]

    gmm_path, point_path = create_visualization_folders(root_dir)

    for (i, gmm) in enumerate(GMMs):
        output_path = f"{gmm_path}/{i}.png"
        plot_gmm_on_image_2d(gmm, image_path, aabb, output_path)

    for (i, iter) in enumerate(valid_samples):
        output_path = f'{point_path}/{i}.png'
        plot_points_in_itration(image_path, iter, aabb, output_path, False, True, nodes)

    # plot_rays_on_image(image_path, aabb, nodes, intersection_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform logs to visualizations")
    parser.add_argument("root_dir", help="Path to the scene root directory")

    args = parser.parse_args()
    main(args.root_dir)
