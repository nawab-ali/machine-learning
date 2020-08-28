#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

""" Deep Learning system for identifying faces. """

import warnings
import argparse
from Image import *
from ImageData import *
from FaceEmbedding import *
from FaceIdentifier import *
from numpy import load
from numpy import savez_compressed
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

class FaceIdentifier(object):
    def __init__(self, face_embeddings_file):
        """
        Initialize FaceIdentifier object.

        Arguments:
        self
        face_embeddings_file -- compressed face embeddings file

        Returns:
        None
        """

        face_embeddings = load(face_embeddings_file)
        self.model = None
        self.train_x = face_embeddings['arr_0']
        self.train_y = face_embeddings['arr_1']
        self.test_x = face_embeddings['arr_2']
        self.test_y = face_embeddings['arr_3']

    def train(self):
        """
        Train FaceIdentifier model.

        Arguments:
        self

        Returns:
        None
        """

        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(self.train_x, self.train_y)

    def predict(self):
        """
        Identify faces using trained FaceIdentifier model.

        Arguments:
        self

        Returns:
        train, test data set prediction accuracy
        """

        yhat_train = self.model.predict(self.train_x)
        yhat_test = self.model.predict(self.test_x)

        return accuracy_score(self.train_y, yhat_train), accuracy_score(self.test_y, yhat_test)

def main():
    warnings.filterwarnings('ignore')

    args = parse_args()
    train_x, train_y, test_x, test_y = load_data(args.training_dir, args.validation_dir)
    print('Image data: ', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    print('Create face embeddings...')
    embedding_train_x, embedding_test_x = get_face_embeddings(args.facenet_model, train_x, test_x)
    norm_embedding_train_x, norm_embedding_test_x = normalize_face_embeddings(embedding_train_x,
                                                                              embedding_test_x)
    encoded_train_y, encoded_test_y = encode_labels(train_y, test_y)
    savez_compressed(args.face_embeddings_file, norm_embedding_train_x, encoded_train_y,
                     norm_embedding_test_x, encoded_test_y)

    print('Train FaceIdentifier model...')
    face_identifier = FaceIdentifier(args.face_embeddings_file)
    face_identifier.train()

    train_accuracy, test_accuracy = face_identifier.predict()
    print('Prediction accuracy: train=%.2f test=%.2f' % (train_accuracy*100, test_accuracy*100))

def load_data(training_dir, validation_dir):
    """
    Load training and validation data.

    Arguments:
    training_dir -- directory with training data
    validation_dir -- directory with validation data

    Returns:
    train_x -- training data
    train_y -- training data label
    test_x -- validation data
    test_y -- validation data label
    """

    print('Load training data...')
    training_data = ImageData(training_dir)
    train_x, train_y = training_data.load_dataset()

    print('Load validation data...')
    validation_data = ImageData(validation_dir)
    test_x, test_y = validation_data.load_dataset()

    return train_x, train_y, test_x, test_y

def get_face_embeddings(facenet_model, train_x, test_x):
    """
    Return face embeddings.

    Arguments:
    facenet_model -- trained FaceNet model
    train_x -- training data
    test_x -- validation data

    Returns:
    embedding_train_x -- embeddings of training data
    embedding_test_x -- embeddings of validation data
    """

    face_embedding = FaceEmbedding(facenet_model)

    embedding_train_x = [face_embedding.get_face_embedding(face) for face in train_x]
    embedding_test_x = [face_embedding.get_face_embedding(face) for face in test_x]

    return embedding_train_x, embedding_test_x

def normalize_face_embeddings(embedding_train_x, embedding_test_x):
    """
    Normalize face embeddings.

    Arguments:
    embedding_train_x -- embeddings of training data
    embedding_test_x -- embeddings of validation data

    Returns:
    Normalized face embedding vectors
    """

    normalizer = Normalizer(norm='l2')
    return normalizer.transform(embedding_train_x), normalizer.transform(embedding_test_x)

def encode_labels(train_y, test_y):
    """
    Encode string labels into integers.

    Arguments:
    train_y -- training data labels
    test_y -- validation data labels

    Returns:
    Encoded data labels
    """

    encoder = LabelEncoder()
    encoder.fit(train_y)

    return encoder.transform(train_y), encoder.transform(test_y)

def parse_args():
    """
    Parse command line arguments.

    Arguments:
    None

    Returns:
    Parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--facenet_model', required=True, help='Trained FaceNet model')
    parser.add_argument('--training_dir', required=True, help='Training images directory')
    parser.add_argument('--validation_dir', required=True, help='Validation images directory')
    parser.add_argument('--face_embeddings_file', required=True, help='Face embeddings file')

    return parser.parse_args()

if __name__ == '__main__':
    main()
