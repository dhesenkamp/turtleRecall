from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision


# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf


def create_compiled_alexnet():
    """
    AlexNet convolutional neural network as described in Krizhevsky et al., 2012.
    Usually works with input of shape (227, 227, 3).
    """

    alexNet = Sequential([
        Input(shape=(224,224,3)),

        Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),

        Flatten(),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=4096, activation='relu'),
        Dropout(rate=0.5),

        Dense(units=254, activation='softmax')
    ])

    alexNet.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy(), Precision(top_k=5)]
)

    return alexNet