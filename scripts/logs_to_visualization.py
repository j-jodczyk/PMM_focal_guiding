from log_parser import parse_log_file
from gmm_on_img import plot_gmm_on_image_2d
from points_on_img import plot_points_in_itration
from rays_on_image import plot_rays_on_image
from octree_on_img import plot_octree_on_image
import os
from datetime import datetime
import argparse
from pathlib import Path

def create_visualization_folders(root):
    date_str = datetime.today().strftime("%d-%m")
    base_dir = Path(f"{root}/visualizations/{date_str}")

    if base_dir.exists():
        counter = 1
        while Path(f"{root}/visualizations/{date_str}_{counter:02d}").exists():
            counter += 1
        base_dir = Path(f"{root}/visualizations/{date_str}_{counter:02d}")

    # final_dir_gmm = base_dir / "gmm"
    # final_dir_gmm.mkdir(parents=True, exist_ok=True)

    # final_dir_points = base_dir / "points"
    # final_dir_points.mkdir(parents=True, exist_ok=True)

    final_dir_octree = base_dir / "octree"
    final_dir_octree.mkdir(parents=True, exist_ok=True)

    # return (str(final_dir_gmm), str(final_dir_points))
    return str(final_dir_octree)


def run(root_dir):
    log_file_path = f"{root_dir}/mitsuba.DESKTOP-06NEMHS.log"
    image_path = f"../scenes/modern-hall/Reference.png"
    GMMs, valid_samples, intersection_data, octrees = parse_log_file(log_file_path)
    aabb = octrees[0]["main_aabb"]
    nodes = octrees[0]["leaf_nodes"]

    octrees = list(filter(lambda tree: (len(tree["leaf_nodes"]) > 0), octrees))
    # gmm_path, point_path = create_visualization_folders(root_dir)
    octree_path = create_visualization_folders(root_dir)

    # not excactly like this:
    converging = octrees[0]
    diverging = octrees[1]

    output_path_converging = f"{octree_path}/converging.png"
    output_path_diverging = f"{octree_path}/diverging.png"

    plot_octree_on_image(image_path, converging, output_path_converging)
    plot_octree_on_image(image_path, diverging, output_path_diverging)

    # for (i, gmm) in enumerate(GMMs):
    #     output_path = f"{gmm_path}/{i}.png"
    #     plot_gmm_on_image_2d(gmm, image_path, aabb, output_path)

    # for (i, iter) in enumerate(valid_samples):
    #     output_path = f'{point_path}/{i}.png'
    #     plot_points_in_itration(image_path, iter, aabb, output_path, False, True, nodes)

    # plot_rays_on_image(image_path, aabb, nodes, intersection_data)

def main(root_dir):
    directory = "/home/julia/PBRT/PMM_focal_guiding/argument_tuning/modern-hall/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.5_treeThreshold_0.0001"
    run(directory)
    # directories = [d.name for d in Path(root_dir).iterdir() if d.is_dir()]
    # for d in directories:
    #     run(f"{root_dir}/{d}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform logs to visualizations")
    parser.add_argument("root_dir", help="Path to the scene root directory")

    args = parser.parse_args()
    main(args.root_dir)
