import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm


def train_step(model, input, target, loss_function, optimizer):

    with tf.GradientTape() as tape:
        prediction = model(input, training=True)
        loss = loss_function(target, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test_step(model, test_data, loss_function):

    #test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        #sample_test_accuracy = 
        test_loss_aggregator.append(sample_test_loss.numpy())
    
    test_loss = tf.reduce_mean(test_loss_aggregator)

    return test_loss


def training_loop(train_data, test_data, model, loss_function, optimizer, nr_epochs):
    tf.keras.backend.clear_session()

    train_losses = []
    test_losses = []

    # testing once before training
    test_loss = test_step(model, test_data, loss_function)
    test_losses.append(test_loss)

    # training
    for epoch in tqdm.trange(nr_epochs, unit='epoch', desc='Training progress: ', postfix=f'Loss {test_losses[-1]}'):
        if epoch % 10 == 0:
            #print(f'Epoch {str(epoch)}: starting with loss {test_losses[-1]}')
            pass
        
        epoch_loss_aggregator = []
        for input, target in train_data:
            train_loss = train_step(model, input, target, loss_function, optimizer)
            epoch_loss_aggregator.append(train_loss)
        
        train_losses.append(tf.reduce_mean(epoch_loss_aggregator))

        test_loss = test_step(model, test_data, loss_function)
        test_losses.append(test_loss)
    
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel(f'{loss_function.name}')
    plt.legend()