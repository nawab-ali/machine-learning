#!/bin/bash

./FaceIdentifier.py \
	--facenet_model data/facenet_keras.h5 \
	--training_dir data/5-celebrity-faces-dataset/train/ \
	--validation_dir data/5-celebrity-faces-dataset/val/ \
	--face_embeddings_file data/5-celebrity-faces-embeddings.npz
