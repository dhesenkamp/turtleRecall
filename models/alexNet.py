from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from util import NUM_CLASSES

class AlexNet(Model):
    """AlexNet convolutional neural network as described in Krizhevsky et al., 2012."""

    def __init__(self):
        super(AlexNet, self).__init__()

        self.all_layers = [
            Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)), #normally 227 x 277
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
            Dense(units=NUM_CLASSES, activation='softmax')
        ]
        
    
    def call(self, x, training=False):
        for layer in self.all_layers:
            try:
                x = layer(x, training=training)
            except:
                x = layer(x)
        return x