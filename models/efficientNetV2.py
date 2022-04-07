import tensorflow as tf
import tensorflow_hub as hub


def create_compiled_EfficientNetV2(trainable=False):
    """
    EfficientNetV2 as per Tan and Le (2021)
    https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2
    https://arxiv.org/abs/2104.00298v2
    """

    NUM_CLASSES = 254

    efficientNet = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),
        hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", trainable=False),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    efficientNet.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(top_k=5)]
    )

    return efficientNet