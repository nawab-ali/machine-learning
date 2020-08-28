#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

import warnings
import unittest
from Image import *
from ImageData import *
from FaceEmbedding import *
from FaceIdentifier import *

class TestImage(unittest.TestCase):
    def test_preprocess_image(self):
        filename = 'data/5-celebrity-faces-dataset/val/ben_affleck/httpcsvkmeuaafdfjpg.jpg'
        image = Image(filename)
        pixels = image._preprocess_image()
        self.assertEqual(pixels.shape, (264, 200, 3))

    def test_detect_face(self):
        filename = 'data/5-celebrity-faces-dataset/val/ben_affleck/httpcsvkmeuaafdfjpg.jpg'
        image = Image(filename)
        pixels = image._preprocess_image()
        x1, y1, x2, y2 = image._detect_face(pixels)
        self.assertEqual([x1, y1, x2, y2], [113, 42, 151, 94])

    def test_extract_face(self):
        filename = 'data/5-celebrity-faces-dataset/val/ben_affleck/httpcsvkmeuaafdfjpg.jpg'
        image = Image(filename)
        face = image.extract_face()
        self.assertEqual(face.shape, (160, 160, 3))

class TestFaceEmbedding(unittest.TestCase):
    def test_normalize_face(self):
        warnings.filterwarnings('ignore')
        filename = 'data/5-celebrity-faces-dataset/val/ben_affleck/httpcsvkmeuaafdfjpg.jpg'
        face_embedding = FaceEmbedding('data/facenet_keras.h5')
        image = Image(filename)
        face = image.extract_face()
        normalized_face = face_embedding._normalize_face(face)
        self.assertEqual(normalized_face.shape, (1, 160, 160, 3))

    def test_get_face_embedding(self):
        warnings.filterwarnings('ignore')
        filename = 'data/5-celebrity-faces-dataset/val/ben_affleck/httpcsvkmeuaafdfjpg.jpg'
        face_embedding = FaceEmbedding('data/facenet_keras.h5')
        image = Image(filename)
        face = image.extract_face()
        embedding = face_embedding.get_face_embedding(face)
        self.assertEqual(embedding.shape, (128,))

class TestImageData(unittest.TestCase):
    @unittest.skip('test takes too long')
    def test_load_faces_from_dir(self):
        directory = 'data/5-celebrity-faces-dataset/val/ben_affleck/'
        image_data = ImageData(directory)
        faces = image_data._load_faces_from_dir(directory)
        self.assertEqual(len(faces), 5)

    @unittest.skip('test takes too long')
    def test_load_dataset(self):
        directory = 'data/5-celebrity-faces-dataset/val/'
        image_data = ImageData(directory)
        X, y = image_data.load_dataset()
        self.assertEqual(X.shape, (25, 160, 160, 3))
        self.assertEqual(y.shape, (25,))

    @unittest.skip('test takes too long')
    def test_load_dataset_without_trailing_slash(self):
        directory = 'data/5-celebrity-faces-dataset/val'
        image_data = ImageData(directory)
        X, y = image_data.load_dataset()
        self.assertEqual(X.shape, (25, 160, 160, 3))
        self.assertEqual(y.shape, (25,))

class TestFaceIdentifier(unittest.TestCase):
    def test_predict(self):
        face_identifier = FaceIdentifier('data/5-celebrity-faces-embeddings.npz')
        face_identifier.train()
        train_accuracy, test_accuracy = face_identifier.predict()
        self.assertEqual(train_accuracy, 1.00)
        self.assertEqual(test_accuracy, 1.00)

if __name__ == '__main__':
    unittest.main()
