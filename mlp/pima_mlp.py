#!/home/ubuntu/wspace-2/anaconda2/bin/python

""" Pima Indians onset of diabetes binary classification problem. """

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def processData(filename):
    """ Load and prepare the dataset. """

    dataset = np.loadtxt(filename, delimiter=',')

    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    return X, Y

def model():
    """ Create the MLP model. """

    mlpModel = Sequential()
    mlpModel.add(Dense(12, input_dim=8, activation='relu'))
    mlpModel.add(Dense(1, activation='sigmoid'))

    return mlpModel

def main():
    X, Y = processData('pima-indians-diabetes.csv')

    # Create the MLP model
    mlpModel = model()

    # Compile the model
    mlpModel.compile(loss='binary_crossentropy', optimizer='adam',
                     metrics=['accuracy'])

    # Train the model
    history = mlpModel.fit(X, Y, epochs=100, batch_size=10, verbose=1)

    # Evaluate the model
    loss, accuracy = mlpModel.evaluate(X, Y)
    print 'Training Loss: ', '{:.4f}'.format(loss)
    print 'Training Accuracy: ', '{:.4f}'.format(accuracy)

    # Make predictions
    probabilities = mlpModel.predict(X)
    predictions = [float(round(x)) for x in probabilities]
    accuracy = np.mean(predictions == Y)
    print 'Inference Accuracy: ', '{:.4f}'.format(accuracy)

if __name__ == '__main__':
    main()
