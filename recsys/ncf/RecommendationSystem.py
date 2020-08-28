#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

""" Neural Collaborative Filtering based Recommendation System for movies, books etc. """

import sys
import argparse
import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append('/home/ubuntu/efs/machine-learning/dl_visualizer/')
from DLVisualizer import DLVisualizer

class RecommendationSystem:
    def __init__(self, training_data_file, trained_model_file, model_plot_file='model_plot.png',
                 model_accuracy_file='model_accuracy.png', model_loss_file='model_loss.png',
                 num_latent_factors=64, validation_split=0.25):
        """
        Initialize Recommendation System object.

        Arguments:
        self
        training_data_file -- file containing training data
        trained_model_file -- file where trained model will be saved
        model_plot_file -- file where model plot will be saved
        model_accuracy_file -- file where model accuracy plot will be saved
        model_loss_file -- file where model loss plot will be saved
        num_latent_factors -- number of latent factors for users and items
        validation_split -- split between training and validation data

        Returns:
        None
        """

        self.num_users, self.num_items, self.train_data, self.test_data = \
                                        self._preprocess_training_data(training_data_file)
        self.trained_model_file = trained_model_file
        self.model_plot_file = model_plot_file
        self.model_accuracy_file = model_accuracy_file
        self.model_loss_file = model_loss_file
        self.num_latent_factors = num_latent_factors
        self.validation_split = validation_split

    def train(self, epochs=100, batch_size=128):
        """
        Train the neural network.

        Arguments:
        self
        epochs -- number of epochs to train
        batch_size -- batch size

        Returns:
        None
        """

        model = self._create_training_model(self.num_users, self.num_items)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        history = model.fit([self.train_data.user_id.values, self.train_data.item_id.values],
                            self.train_data.rating.values, validation_split=self.validation_split,
                            epochs=epochs, batch_size=batch_size)
        model.save(self.trained_model_file)

        dl_visualizer = DLVisualizer(model, history)
        dl_visualizer.visualize_model(self.model_plot_file)
        dl_visualizer.visualize_model_loss(self.model_loss_file)
        dl_visualizer.visualize_model_accuracy(self.model_accuracy_file)

    def recommend(self):
        """
        Make recommendations based on the trained neural network.

        Arguments:
        self

        Returns:
        Accuracy
        Predicted recommendations for [user, item] tuple
        """

        model = load_model(self.trained_model_file)
        return np.round(model.predict([self.test_data.user_id.values,
                        self.test_data.item_id.values]))

    def _preprocess_training_data(self, training_data_file):
        """
        Read the training dataset from file and preprocess it.

        Arguments:
        self
        training_data_file -- file containing training data

        Returns:
        Pandas dataframes containing train and test datasets
        """

        dataset = pd.read_csv(training_data_file, sep='\t',
                              names='user_id,item_id,rating,timestamp'.split(','))

        # Assign a unique user_id between (0, #users)
        dataset.user_id = dataset.user_id.astype('category').cat.codes.values
        # Assign a unique item_id between (0, #items)
        dataset.item_id = dataset.item_id.astype('category').cat.codes.values

        num_users = len(dataset.user_id.unique())
        num_items = len(dataset.item_id.unique())
        train_data, test_data = train_test_split(dataset, test_size=0.2)

        return num_users, num_items, train_data, test_data

    def _create_training_model(self, num_users, num_items, dropout=0.2):
        """
        Create a Keras training model.

        Arguments:
        self
        num_users -- number of users
        num_items -- number of items
        dropout -- dropout value

        Returns:
        model -- Keras training model
        """

        user_input, user_vector = self._create_embedding('user', num_users, self.num_latent_factors)
        item_input, item_vector = self._create_embedding('item', num_items, self.num_latent_factors)

        concat_vector = concatenate([user_vector, item_vector], name='concat')
        dropout_1 = Dropout(dropout, name='dropout_1')(concat_vector)
        dense_1 = Dense(200, name='fully_connected_1')(dropout_1)
        dropout_2 = Dropout(dropout, name='dropout_2')(dense_1)
        dense_2 = Dense(100, name='fully_connected_2')(dropout_2)
        dropout_3 = Dropout(dropout, name='dropout_3')(dense_2)
        dense_3 = Dense(50, name='fully_connected_3')(dropout_3)
        dropout_4 = Dropout(dropout, name='dropout_4')(dense_3)
        dense_4 = Dense(20, name='fully_connected_4', activation='relu')(dropout_4)
        result = Dense(1, name='activation', activation='relu')(dense_4)

        return Model([user_input, item_input], result)

    def _create_embedding(self, name, input_dim, output_dim, dropout=0.2):
        """
        Create a Keras embedding.

        Arguments:
        self
        name -- embedding name
        input_dim -- input dimension
        output_dim -- output dimension
        dropout -- dropout value

        Returns:
        input_vector -- input vector
        output_vector -- embedding vector
        """

        input_vector = Input(shape=[1], name=name+'_input')
        embedding = Embedding(input_dim+1, output_dim, input_length=1,
                              name=name+'_embedding')(input_vector)
        output_vector = Flatten(name=name+'_flatten')(embedding)
        output_vector = Dropout(dropout, name=name+'_dropout')(output_vector)

        return input_vector, output_vector

def main():
    """
    Main thread of execution.

    Arguments:
    None

    Returns:
    None
    """

    args = parse_args()
    recommendation_system = RecommendationSystem(training_data_file=args.training_data,
                                                 trained_model_file=args.trained_model_file,
                                                 model_plot_file=args.model_plot_file,
                                                 model_accuracy_file=args.model_accuracy_file,
                                                 model_loss_file=args.model_loss_file,
                                                 num_latent_factors=args.num_latent_factors,
                                                 validation_split=args.validation_split)

    if args.mode == 'training':
        recommendation_system.train(epochs=args.epochs, batch_size=args.batch)
    else:
        recommendations = recommendation_system.recommend().flatten()
        mse = mean_squared_error(recommendation_system.test_data.rating.values, recommendations)
        print('Recommendation Accuracy: %.2f' % mse)

def parse_args():
    """
    Parse command line arguments.

    Arguments:
    None

    Returns:
    Parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['training', 'inference'],
                        help='Deep Learning mode')
    parser.add_argument('--training_data', required=True, help='Training data file')
    parser.add_argument('--trained_model_file', required=True, help='Trained model file')
    parser.add_argument('--model_plot_file', default='model_plot.png', help='Model plot file')
    parser.add_argument('--model_accuracy_file', default='model_accuracy.png',
                        help='Model accuracy file')
    parser.add_argument('--model_loss_file', default='model_loss.png', help='Model loss file')
    parser.add_argument('--num_latent_factors', type=int, default=64, help='Number of latent factors')
    parser.add_argument('--validation_split', type=float, default=0.25, help='Validation split')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size')

    return parser.parse_args()

if __name__ == '__main__':
    main()
