import os
import sys
import flip_evaluator as flip
import matplotlib.pyplot as plt

SCENE = sys.argv[1] # example dining-room

reference_path = f"/home/julka/PMM_focal_guiding/exr_references/{SCENE}_ref.exr"
exr_path = f"{SCENE}.exr"
filp_result_path = f"{SCENE}_flip.png"
rmse_result_path = f"{SCENE}_rmse.png"

print("-------- FLIP ANALYSIS --------")
flipErrorMap, meanFLIPError, parameters = flip.evaluate(reference_path, exr_path, "HDR")
plt.imshow(flipErrorMap)
plt.axis('off')
plt.savefig(filp_result_path, bbox_inches='tight',transparent=True, pad_inches=0)
print(meanFLIPError)

print("-------- RMSE ANALYSIS --------")
rmse_bash_command = f"compare -metric RMSE {exr_path} {reference_path} {rmse_result_path}"
os.system(rmse_bash_command)
