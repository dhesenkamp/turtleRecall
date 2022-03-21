import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from util import IMAGE_SIZE

def crop_and_resize(pil_img):
  """Crops square from center of image and resizes."""

  w, h = pil_img.size
  crop_size = min(w, h)
  crop = pil_img.crop(((w - crop_size) // 2, (h - crop_size) // 2,
                       (w + crop_size) // 2, (h + crop_size) // 2))
  
  return crop.resize(IMAGE_SIZE)


