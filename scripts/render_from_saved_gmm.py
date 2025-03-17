import os
import shutil
import xml.etree.ElementTree as ET
import subprocess
import itertools

saved_gmms = [
    ("modern-hall", "../argument_tuning/modern-hall/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.5_treeThreshold_0.01", 0),
    ("modern-hall", "../argument_tuning/modern-hall/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.5_treeThreshold_0.001", 1),
    ("living-room", "../argument_tuning/living-room/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.5", 0),
    ("living-room", "../argument_tuning/living-room/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.7", 1),
    ("dining-room", "../argument_tuning/dining-room/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.5", 0),
    ("dining-room", "../argument_tuning/dining-room/gmmSplittingThreshold_4.0_gmmMergingThreshold_0.7", 1),
]

# Read the XML file
def load_xml(file_path):
    tree = ET.parse(file_path)
    return tree, tree.getroot()

# Modify the XML based on parameters
def modify_xml(file_path, value):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for elem in root.findall(f".//string[@name='importGmmFile']"):
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
def move_results(output_folder, scene_name, num):
    os.makedirs(output_folder, exist_ok=True)
    log_path = "../mitsuba/mitsuba.DESKTOP-06NEMHS.log"
    log_file = "mitsuba.DESKTOP-06NEMHS"
    scene_path = f"../scenes/{scene_name}/{scene_name}.exr"
    shutil.move(scene_path, os.path.join(output_folder, f"{scene_name}_{num}.exr"))
    shutil.move(log_path, os.path.join(output_folder, f"{log_file}_{num}.log"))

# Main function
def main():
    xml_file = "../scenes/_integrators/pmm_focal_guiding.xml"
    for (scene_name, gmm, num) in saved_gmms:
        scene_file = f"../../scenes/{scene_name}/{scene_name}.xml" # relative to mitsuba dir
        output_base_dir = f"../256spp/{scene_name}"

        tree = modify_xml(xml_file, f"{gmm}/final_gmm.serialized")
        save_xml(tree, xml_file)

        output_folder = os.path.join(output_base_dir)

        run_mitsuba(scene_file)

        # Move logs and output files
        move_results(output_folder, scene_name, num)

        print(f"Results saved in {output_folder}")

    print("All tests completed!")

if __name__ == "__main__":
    main()
