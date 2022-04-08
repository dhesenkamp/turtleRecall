import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def train_step(model, input, target, loss_function, optimizer):
    """
    Utility function to use with training of sub-classed models. Trains model on given data.

    Args:
        model (tf.keras.Model): model instance to train
        input: training data
        target: to training data corresponding labels
        loss_function (tf.keras.losses object): loss function to calculate loss of the model
        optimizer (tf.keras.optimizer object): optimizer to use for training the model
    Returns:
        loss (float): combined loss of the model after training on the given data
    """
    with tf.GradientTape() as tape:
        prediction = model(input, training=True)
        loss = loss_function(target, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test_step(model, test_data, loss_function):
    """
    Utility function to use with training of sub-classed models. Performs model evaluation on test data.

    Args: 
        model (tf.keras.Model): model instance to test
        test_data (tf.data.Dataset): dataset for testing
        loss_function (tf.keras.losses object): loss function to calculate loss of the model
    Returns:
        test_loss (float): mean loss of the model on the test set
    """

    #test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        #sample_test_accuracy = 
        test_loss_aggregator.append(sample_test_loss.numpy())
    
    test_loss = tf.reduce_mean(test_loss_aggregator)

    return test_loss


def training_loop(train_data, test_data, model, loss_function, optimizer, nr_epochs, plot=True):
    """
    Utility function to train sub-classed models. Loops nr_epochs times over the given dataset.

    Args:
        train_data (tf.data.Dataset): dataset for training
        test_data (tf.data.Dataset): dataset for testing
        model (tf.keras.Model): model instance to train
        loss_function (tf.keras.losses object): loss function to use for backpropagation        
        optimizer (tf.keras.optimizer object): optimizer to use for training the model
        nr_epochs (int): number of epochs to train for
        plot (Bool): whether to visualize results of the training process in a graph. Default: True
    Returns:
        None
    """

    tf.keras.backend.clear_session()

    train_losses = []
    test_losses = []

    # testing once before training
    test_loss = test_step(model, test_data, loss_function)
    test_losses.append(test_loss)

    # training
    for epoch in trange(nr_epochs, unit='epoch', desc='Training progress: ', postfix=f'Loss {test_losses[-1]}'):
        
        epoch_loss_aggregator = []
        for input, target in tqdm(train_data):
            train_loss = train_step(model, input, target, loss_function, optimizer)
            epoch_loss_aggregator.append(train_loss)
        
        train_losses.append(tf.reduce_mean(epoch_loss_aggregator))

        test_loss = test_step(model, test_data, loss_function)
        test_losses.append(test_loss)
    
    # return plot with training metrics if desired
    if plot:
        plt.figure()
        plt.plot(train_losses, label='train loss')
        plt.plot(test_losses, label='test loss')
        plt.xlabel('epoch')
        plt.ylabel(f'{loss_function.name}')
        plt.legend()