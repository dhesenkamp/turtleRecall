import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns


IMAGE_DIR = './data/images'
MIN_NR_IMGS = 10
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)
NUM_CLASSES = 254
BATCH_SIZE = 128
NR_EPOCHS = 50

