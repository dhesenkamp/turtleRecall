from tensorflow_hub import KerasLayer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision
from utils import NUM_CLASSES, INPUT_SHAPE


def create_compiled_EfficientNetV2(trainable=True):
    """
    EfficientNetV2-B0 as per Tan and Le (2021)
    https://arxiv.org/abs/2104.00298v2

    Args:
        trainable (bool): whether the imported model's parameters are fine-tunable. Default: True
    Returns:
        model (Tensorflow model): compiled model, ready to be trained
    """

    hub_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"

    model = Sequential([
        InputLayer(input_shape=INPUT_SHAPE),
        KerasLayer(hub_url, trainable=trainable),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(), Precision(top_k=5)]
    )

    return model