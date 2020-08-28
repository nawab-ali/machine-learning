#!/home/ubuntu/wspace-2/anaconda2/bin/python

"""
Train and evaluate an MLP on the Reuters newswire topic classification task.
"""

import keras
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer

num_epochs = 10
batch_size = 32
max_words = 1000

def processData():
    """ Pre-process the Reuters data. """

    (x_train,y_train), (x_test,y_test) = reuters.load_data(num_words=max_words,
                                                           test_split=0.2)

    # Tokenize the data
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    # Convert class vector to binary class matrix
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes

def model(num_classes):
    """ Create the MLP model. """

    mlpModel = Sequential()
    mlpModel.add(Dense(512, input_shape=(max_words,), activation='relu'))
    mlpModel.add(Dropout(0.5))
    mlpModel.add(Dense(num_classes, activation='softmax'))

    return mlpModel

def main():
    # Pre-process the Reuters data
    x_train, y_train, x_test, y_test, num_classes = processData()

    # Train the model
    mlpModel = model(num_classes)
    mlpModel.compile(loss='categorical_crossentropy', optimizer='adam',
                     metrics=['accuracy'])
    mlpModel.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                 verbose=1, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = mlpModel.evaluate(x_test, y_test, batch_size=batch_size,
                                       verbose=1)

    print 'Loss: ', '{:.4f}'.format(loss)
    print 'Accuracy: ', '{:.4f}'.format(accuracy)

if __name__ == '__main__':
    main()
