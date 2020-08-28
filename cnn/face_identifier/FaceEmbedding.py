""" Deep Learning system for creating face embeddings. """

import numpy as np
from keras.models import load_model

class FaceEmbedding(object):
    def __init__(self, facenet_model):
        """
        Initialize FaceEmbedding object.

        Arguments:
        self
        facenet_model -- pre-trained FaceNet model

        Returns:
        None
        """

        self.facenet_model = load_model(facenet_model)

    def get_face_embedding(self, face):
        """
        Return face embedding.

        Arguments:
        self
        face -- pixels of face

        Returns:
        Face embedding
        """

        normalized_face = self._normalize_face(face)
        yhat = self.facenet_model.predict(normalized_face)
        return yhat[0]

    def _normalize_face(self, face):
        """
        Normalize face for embedding.

        Arguments:
        self
        face -- pixels of face

        Returns:
        Normalized face pixels
        """

        face = face.astype('float32')
        mean, stddev = face.mean(), face.std()
        face = (face-mean)/stddev
        return np.expand_dims(face, axis=0)
