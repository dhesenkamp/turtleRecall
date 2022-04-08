from tensorflow_hub import KerasLayer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision
from utils import NUM_CLASSES


def create_compiled_EfficientNetV2(trainable=True):
    """
    EfficientNetV2 as per Tan and Le (2021)
    https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2
    https://arxiv.org/abs/2104.00298v2
    """

    hub_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"
    
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