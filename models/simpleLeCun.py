import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten


class LeCun(tf.keras.Model):

    def __init__(self):

        super(LeCun, self).__init__()

        self.all_layers = [
            Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2)),
            
            Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2)),

            Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2)),

            Flatten(),
            Dense(units=64),
            Activation('relu'),
            Dropout(rate=0.5),
            Dense(units=1),
            Activation('sigmoid')
        ]

    
    def call(self, x, training=False):

        for layer in self.all_layers:
            try:
                x = layer(x, training=training)
            except:
                x = layer(x)

        return x


def create_LeCun():
    """
    Taken from Chollet's blog post on image classification:
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
