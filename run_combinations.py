import xml.etree.ElementTree as ET
import subprocess
import os

# Configuration
xml_path = "/home/julka/PMM_focal_guiding/mitsuba/scene/_integrators/pmm_focal_recursive.xml"
dockerfile_dir = "."
run_script = "./run_all_scenes.sh"

# (splitting_threshold, merging_threshold, init_method, folder)
combinations = [
    # different init methods
    # (100, 0.65, "KMeans", "init_method/kmeans"),
    # (100, 0.65, "Random", "init_method/random"),
    (100, 0.65, "Uniform", "init_method/uniform"),
    # # # different merging thresholds
    # (100, 0.5, "KMeans", "merging_threshold/0.5"),
    # (100, 0.9, "KMeans", "merging_threshold/0.9"),
    # # different splitting thresholds
    # (1000, 0.65, "KMeans", "splitting_threshold/1000"),
    # (10000, 0.65, "KMeans", "splitting_threshold/10000"),
]

def modify_xml(split_val, merge_val, init_method):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for default in root.findall("default"):
        if default.get("name") == "gmmSplittingThreshold":
            default.set("value", str(split_val))
        elif default.get("name") == "gmmMergingThreshold":
            default.set("value", str(merge_val))

    integrator = root.find("integrator")
    for string in integrator.findall("string"):
        if string.get("name") == "gmm.initMethod":
            string.set("value", init_method)

    tree.write(xml_path)

def build_docker():
    print("Building Docker container...")
    subprocess.run(["docker", "build", dockerfile_dir, "-t", "pmm_focal_guiding"], check=True)

def run_bash_script(extra_arg):
    print(f"Running bash script saving results to: {extra_arg}")
    subprocess.run([run_script, extra_arg], check=True)

def main():
    for split, merge, init, extra in combinations:
        print(f"\n--- Running combination: Split={split}, Merge={merge}, Init={init}, Arg={extra} ---")
        modify_xml(split, merge, init)
        build_docker()
        run_bash_script(extra)

if __name__ == "__main__":
    main()
