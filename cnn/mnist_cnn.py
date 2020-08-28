#!/home/ubuntu/wspace-2/anaconda2/bin/python

""" Implement a simple CNN in Keras. """

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

def preProcessData():
    """ Pre-process the training and test datasets. """

    # Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape tensors to include 1 input channel
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Convert data type and normalize values
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert 1-dim class tensors to 10-dim class tensors
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test

def model():
    """ Define a simple CNN model architecture. """

    model = Sequential()

    # CNN input layer
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                     input_shape=(28, 28, 1)))

    # Stack the other layers
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='softmax'))

    return model

def main():
    X_train, Y_train, X_test, Y_test = preProcessData()

    # Create a CNN model
    cnnModel = model()

    # Compile the model
    cnnModel.compile(loss='categorical_crossentropy', optimizer='adam',
                     metrics=['accuracy'])

    # Fit the Keras model
    cnnModel.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    # Evaluate model on test data
    score = cnnModel.evaluate(X_test, Y_test, verbose=0)
    print 'Test loss: ', '{:.4f}'.format(score[0])
    print 'Test accuracy: ', '{:.4f}'.format(score[1])

if __name__ == '__main__':
    main()
