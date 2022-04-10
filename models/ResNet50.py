import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


def create_resnet50(output_classes):
    """Create pre-trained ResNet50 model with custom top layer."""
    
    inputs = tf.keras.Input(shape=(224,224,3))
    x = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')(inputs)
    outputs = tf.keras.layers.Dense(units=output_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
