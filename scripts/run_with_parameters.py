import os
import shutil
import xml.etree.ElementTree as ET
import subprocess
import argparse
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="Run Mitsuba with different XML configurations.")
    parser.add_argument("scene_name", help="Name of the scene")
    return parser.parse_args()

# Read the XML file
def load_xml(file_path):
    tree = ET.parse(file_path)
    return tree, tree.getroot()

# Modify the XML based on parameters
def modify_xml(file_path, param_values):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for param_name, value in param_values.items():
        for elem in root.findall(f".//default[@name='{param_name}']"):
            elem.set("value", str(value))

    return tree

# Save modified XML to a temporary file
def save_xml(tree, file_path):
    tree.write(file_path)

# Run Mitsuba with the modified XML
def run_mitsuba(scene_file):
    # Build the command to run Mitsuba
    cmd = f"cd ../mitsuba && source setpath.sh && mitsuba {scene_file}"

    # Run Mitsuba in the correct directory
    subprocess.run(cmd, shell=True, executable="/bin/bash")

# Move output files to results directory
def move_results(output_folder, scene_name):
    os.makedirs(output_folder, exist_ok=True)
    log_path = "../mitsuba/mitsuba.DESKTOP-06NEMHS.log"
    log_file = "mitsuba.DESKTOP-06NEMHS.log"
    # serialization_path = f"../scenes/{scene_name}/final_gmm.serialized"
    # serialization_file = "final_gmm.serialized"
    scene_path = f"../scenes/{scene_name}/{scene_name}.exr"
    shutil.move(log_path, os.path.join(output_folder, log_file))
    shutil.move(scene_path, os.path.join(output_folder, f"{scene_name}.exr"))
    # shutil.move(serialization_path, os.path.join(output_folder, serialization_file))

# Main function
def run(scene_name):
    # args = parse_args()
    # scene_name = args.scene_name
    scene_file = f"../../scenes/{scene_name}/{scene_name}.xml" # relative to mitsuba dir

    xml_file = "../scenes/_integrators/pmm_focal_guiding.xml"
    output_base_dir = f"../argument_tuning/{scene_name}"

    # Parameter variations
    # trainingIterations = [5, 10, 15]
    param_ranges = {
        # "gmmSplittingThreshold": [1.0, 4.0, 7.0],
        # "gmmMergingThreshold": [0.3, 0.5, 0.7, 0.9],
        "gmmSplittingThreshold": [4.0],
        "gmmMergingThreshold": [0.5],
        "treeThreshold": [1e-3],
        "gmmInitKMeans": ["false", "true"],
        # "gmmMinNumComp": [3, 5, 8], // not so important now
        # "gmmMaxNumComp": [10, 15, 20],
    }

    # param_keys, param_values = zip(*param_ranges.items())
    # for values in itertools.product(*param_values):
    # param_set = dict(zip(param_keys, values))
    param_sets = [
        {"trainingIterations": 15}
    ]
    for param_set in param_sets:
        print(f"Testing {param_set}")

        tree = modify_xml(xml_file, param_set)
        save_xml(tree, xml_file)

        param_str = "_".join(f"{k}_{v}" for k, v in param_set.items())
        output_folder = os.path.join(output_base_dir, param_str)
        # output_folder = output_base_dir

        run_mitsuba(scene_file)

        # Move logs and output files
        move_results(output_folder, scene_name)

        print(f"Results saved in {output_folder}")

    print("All tests completed!")

def main():
    scenes = ["dining-room", "modern-hall", "living-room"]
    for scene_name in scenes:
        try:
            run(scene_name)
        except:
            continue
        # scene_file = f"../../scenes/{scene_name}/{scene_name}.xml"
        # output_folder = f"../after_major_correction/{scene_name}"
        # run_mitsuba(scene_file)
        # move_results(output_folder, scene_name)
        # print(f"Results saved in {output_folder}")


if __name__ == "__main__":
    main()
