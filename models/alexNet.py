from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout

# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf


class AlexNet(Model):
    """
    AlexNet convolutional neural network as described in Krizhevsky et al., 2012.
    Works with input of shape (227, 227, 3).
    """

    def __init__(self):
        super(AlexNet, self).__init__()

        self.input_layer = Input(shape=(224,224,3))
        self.all_layers = [
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
        ]
        self.out = self.call(self.input_layer)
        
    
    def call(self, x, training=False):
        for layer in self.all_layers:
            try:
                x = layer(x, training=training)
            except:
                x = layer(x)
        return x


def create_alexnet():
    pass