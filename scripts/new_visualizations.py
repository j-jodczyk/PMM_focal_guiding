import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm
from PIL import Image

from log_parser import parse_log_file
from gmm_on_img import plot_gmm_on_image_2d

log_file = sys.argv[1]
image_path =  sys.argv[2]
GMMs, *_, AABB = parse_log_file(log_file)

gmm = GMMs[-1]

plot_gmm_on_image_2d(gmm, image_path, AABB, '', True)