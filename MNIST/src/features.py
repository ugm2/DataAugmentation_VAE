img_rows, img_cols, img_chns = 28, 28, 1
filters = 64
num_conv = 3
batch_size = 100
latent_dim = 8
intermediate_dim = 128
epsilon_std = 1.0
epochs = 200
images_count_training = 4
images_count_test = 1
TOTAL_IMAGES = 5

import keras
from mpl_toolkits.mplot3d import axes3d, Axes3D
from keras import utils as np_utils
from sklearn.cross_validation import train_test_split
import os
import numpy as np
import random
from datetime import datetime
import time
import re
import cv2
random.seed(42)
np.random.seed(42)