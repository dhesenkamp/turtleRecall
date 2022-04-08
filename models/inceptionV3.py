from pickletools import optimize
from tensorlfow.keras import Sequential
from tensorflow_hub import KerasLayer
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision
from utils import NUM_CLASSES


def create_compiled_inceptionV3(trainable=True):
    """
    InceptionV3 CNN as per Szegedy et al. (2015)
    https://arxiv.org/abs/1512.00567
    """

    hub_url = "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"
    
    model = Sequential([
        InputLayer(input_shape=(224,224,3)),
        KerasLayer(hub_url, trainable=trainable),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(), Precision(top_k=5)]
    )

    return model
