from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten


def create_compiled_LeCun():
    """
    LeCun convolutional network. Adapted from Chollet's blog post on image classification:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    Expects input shape of 150x150x3, adapt preprocessing accordingly.
    """

    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(3, 150, 150)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    
    return model
