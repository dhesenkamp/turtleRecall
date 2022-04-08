import tensorflow as tf
import tensorflow_addons as tfa
from numpy import pi


def rotate_images(ds):
    """
    Rotate images by 90, 180, and 270 degrees. Quadruple size of dataset.

    Args:
        ds (TensorFlow dataset): image dataset on which to perform the augmentation
    Returns:
        ds (TensorFlow dataset): augmented dataset
    """

    ds_rotated_90 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=0.5*pi), y))
    ds_rotated_180 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=pi), y))
    ds_rotated_270 = ds.map(lambda x,y: (tfa.image.rotate(x, angles=1.5*pi), y))

    ds = ds_rotated_90.concatenate(ds_rotated_180).concatenate(ds_rotated_270)

    return ds


def apply_mean_filter(ds, filter_shape=7):
    """
    Perform mean filtering on images. Replace image values by mean of neighbouring values,
    effectively introducing a blur and reducing sharpness of the image.
    
    Args:
        ds (TensorFlow dataset): image dataset on which to perform the augmentation
        filter_shape (int): size of the filter with which to perform the convolution. Default: 7 (7x7 filter size)
    Returns:
        ds (TensorFlow dataset): augmented dataset
    """

    ds_mean_filtered = ds.map(lambda x,y: (tfa.image.mean_filter2d(x, filter_shape=filter_shape), y))
    
    return ds_mean_filtered


def apply_gaussian_filter(ds, filter_shape=7, sigma=2.0):
    """
    Apply a Gaussian image blur. Replace image values by neighbouring values, weighted by a Gaussian function.
    Double the size of the input dataset.

    Args:
        ds (TensorFlow dataset): image dataset on which to perform the augmentation
        filter_shape (int): size of the filter with which to perform the convolution. Default: 7 (7x7 filter size)
        sigma (float): standard deviation of the Gaussian function in both x and y direction. Default: 2.0
    Returns:
        ds (TensorFlow dataset): augmented dataset
    """

    ds_gaussian = ds.map(lambda x,y: (tfa.image.gaussian_filter2d(x, filter_shape=filter_shape, sigma=sigma), y))
    
    return ds_gaussian


def random_hsv(ds):
    """
    Randomly adjust hue, saturation, value of an RGB image in the YIQ color space.

    Args:
        ds (TensorFlow dataset): image dataset on which to perform the augmentation
    Returns:
        ds (TensorFlow dataset): augmented dataset
    """

    ds_hsv = ds.map(lambda x,y: (tfa.image.random_hsv_in_yiq(x, max_delta_hue=0.8, lower_saturation=0.2, upper_saturation=0.8, lower_value=0.2, upper_value=0.8), y))
    
    return ds_hsv


def add_noise(ds, sd=0.2):
    """
    Add randomly sampled noise to image values. Clip afterwards to ensure values stay in range [0,1].
    Sample from normal distribution.

    Args:
        ds (TensorFlow dataset): image dataset on which to perform the augmentation
        sd (float): standard deviation of the normal distribution. Higher values = more noise, Default: 0.2
    Returns:
        ds (TensorFlow dataset): augmented dataset
    """

    ds_noise = ds.map(lambda x,y: (x + tf.random.normal(x.shape, mean=0.0, stddev=sd, dtype=tf.float32), y))
    ds_noise = ds_noise.map(lambda x,y: (tf.clip_by_value(x, 0.0, 1.0), y))

    return ds_noise
