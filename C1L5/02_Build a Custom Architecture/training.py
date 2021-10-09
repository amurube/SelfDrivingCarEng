import argparse
import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

from utils import get_datasets, get_module_logger, display_metrics


def create_network(type_NN):
    
    if type_NN == 'LeNet':
        net = tf.keras.models.Sequential()
        net.add(Conv2D(filters = 6, kernel_size= 5, ))
        net.add(AveragePooling2D(pool_size = (2,2), strides = (2,2)))
        net.add(Conv2D(filters = 16, kernel_size= 5))
        net.add(AveragePooling2D(pool_size = (2,2), strides = (2,2)))
        net.add(Flatten())
        net.add(Dense(120, activation = 'relu'))
        net.add(Dense(84, activation = 'relu'))
        net.add(Dense(43, activation = 'softmax'))
        # IMPLEMENT THIS FUNCTION
        
    if type_NN == 'VGG-9':
        net = tf.keras.models.Sequential()
        net.add(Conv2D(64,(3,3), padding = 'same'))
        net.add(Conv2D(64,(3,3), padding = 'same'))
        net.add(MaxPooling2D(pool_size = (2,2), strides = (1,1)))
        net.add(Conv2D(128,(3,3), padding = 'same'))
        net.add(Conv2D(128,(3,3), padding = 'same'))
        net.add(MaxPooling2D(pool_size = (2,2), strides = (1,1)))
        net.add(Flatten())
        net.add(Dense(120, activation = 'relu'))
        net.add(Dense(84, activation = 'relu'))
        net.add(Dense(43))
        
    return net


if __name__  == '__main__':
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-d', '--imdir', required=True, type=str,
                        help='data directory')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='Number of epochs')
    args = parser.parse_args()    

    logger.info(f'Training for {args.epochs} epochs using {args.imdir} data')
    # get the datasets
    train_dataset, val_dataset = get_datasets(args.imdir)

    #type_NN = 'LeNet'
    
    type_NN = 'VGG-9'
    model = create_network(type_NN)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
   
    history = model.fit(x=train_dataset, 
                        epochs=args.epochs, 
                        validation_data=val_dataset)
    display_metrics(history)
   