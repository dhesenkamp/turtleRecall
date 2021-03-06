from tensorflow_hub import KerasLayer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision
from utils import NUM_CLASSES, INPUT_SHAPE


def create_compiled_inceptionV3(trainable=True):
    """
    InceptionV3 CNN as per Szegedy et al. (2015)
    https://arxiv.org/abs/1512.00567

    Args:
        trainable (bool): whether the imported model's parameters are fine-tunable. Default: True
    Returns:
        model (Tensorflow model): compiled model, ready to be trained
    """

    hub_url = "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"

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
