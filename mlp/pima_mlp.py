#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

""" Pima Indians onset of diabetes binary classification problem. """

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def process_data(filename):
    """ Load and prepare the dataset. """

    dataset = np.loadtxt(filename, delimiter=',')

    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    return X, Y

def model():
    """ Create the MLP model. """

    mlp_model = Sequential()
    mlp_model.add(Dense(12, input_dim=8, activation='relu'))
    mlp_model.add(Dense(1, activation='sigmoid'))

    return mlp_model

def main():
    X, Y = process_data('pima-indians-diabetes.csv')

    # Create the MLP model
    mlp_model = model()

    # Compile the model
    mlp_model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

    # Train the model
    history = mlp_model.fit(X, Y, epochs=100, batch_size=10, verbose=1)

    # Evaluate the model
    loss, accuracy = mlp_model.evaluate(X, Y)
    print('Training Loss: ', '{:.4f}'.format(loss))
    print('Training Accuracy: ', '{:.4f}'.format(accuracy))

    # Make predictions
    probabilities = mlp_model.predict(X)
    predictions = [float(np.round(x)) for x in probabilities]
    accuracy = np.mean(predictions == Y)
    print('Inference Accuracy: ', '{:.4f}'.format(accuracy))

if __name__ == '__main__':
    main()
