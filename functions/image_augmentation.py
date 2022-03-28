import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def rotate_images(ds):
    """
    Rotates images by 90, 180, and 270 degrees.
    Quadruples size of dataset.
    """

    ds_rotated_90 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=0.5*np.pi), y))
    ds_rotated_180 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=np.pi), y))
    ds_rotated_270 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=1.5*np.pi), y))

    ds = ds.concatenate(ds_rotated_90).concatenate(ds_rotated_180).concatenate(ds_rotated_270)

    return ds


def apply_mean_filter(ds, filter_shape):
    """
    Perform mean filtering on images. Replace image values by mean of neighbouring values,
    effectively introducing a blur and reducing sharpness of the image.
    
    Args:
    Returns:
    """

    ds_mean_filtered = ds.map(lambda x,y: (tfa.image.mean_filter2d(x, filter_shape=filter_shape), y))
    ds = ds.concatenate(ds_mean_filtered)
    
    return ds


def apply_gaussian_filter(ds, filter_shape=7, sigma=2):
    """
    Apply a Gaussian image blur. Doubles the size of the input dataset.
    """

    ds_gaussian = ds.map(lambda x,y: (tfa.image.gaussian_filter2d(x, filter_shape=filter_shape, sigma=sigma), y))
    ds = ds.concatenate(ds_gaussian)

    return ds


def random_hsv(ds):
    """
    Randomly adjust hue, saturation, value of an RGB image in the YIQ color space.
    """

    ds_hsv = ds.map(lambda x,y: (tfa.image.random_hsv_in_yiq(x, max_delta_hue=0.8, lower_saturation=0.2, upper_saturation=0.8, lower_value=0.2, upper_value=0.8), y))
    ds = ds.concatenate(ds_hsv)

    return ds


def add_noise(ds, sd=0.3):
    """
    Additive noise
    """

    ds_noise = ds.map(lambda x,y: (x + tf.random.normal(x.shape, mean=0.0, stddev=sd, dtype=tf.float32), y))
    ds_noise = ds_noise.map(lambda x,y: (tf.clip_by_value(x, 0.0, 1.0), y))
    ds = ds.concatenate(ds_noise)

    return ds
