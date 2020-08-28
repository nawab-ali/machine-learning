#!/bin/bash

#./RecommendationSystem.py --mode training --training_data data/movielens_100k.data --trained_model_file data/movielens_100k.model
./RecommendationSystem.py --mode inference --training_data data/movielens_100k.data --trained_model_file data/movielens_100k.model
