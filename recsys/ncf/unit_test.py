#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

import unittest
from RecommendationSystem import *

class TestRecommendationSystem(unittest.TestCase):
    def test_preprocess_training_data_train_data_shape(self):
        recommendation_system = RecommendationSystem('data/movielens_100k.data', None)
        self.assertEqual(recommendation_system.train_data.shape, (80000, 4))

    def test_preprocess_training_data_test_data_shape(self):
        recommendation_system = RecommendationSystem('data/movielens_100k.data', None)
        self.assertEqual(recommendation_system.test_data.shape, (20000, 4))

    def test_train(self):
        recommendation_system = RecommendationSystem(training_data_file='data/movielens_100k.data',
                                                     trained_model_file='/tmp/movielens_100k.model',
                                                     model_plot_file='/tmp/model_plot.png',
                                                     model_accuracy_file='/tmp/model_accuracy.png',
                                                     model_loss_file='/tmp/model_loss.png')
        recommendation_system.train(epochs=10, batch_size=128)

if __name__ == '__main__':
    unittest.main()
