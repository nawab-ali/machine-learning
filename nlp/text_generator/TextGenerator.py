#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

""" Generate text sequences using a statistical language model. """

import sys
import argparse
from pickle import load
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

sys.path.append('/home/ubuntu/efs/machine-learning/nlp/text_preprocessor/')
from TextPreProcessor import TextPreProcessor

class TextGenerator:
    def __init__(self, training_data_file, trained_model_file, word2int_map_file,
                 sequence_length=51):
        """
        Initialize text generator object.

        Arguments:
        self
        training_data_file -- file containing training data
        trained_model_file -- file where trained model will be saved
        word2int_map_file -- file containing words to integers mapping
        sequence_length -- length of sequence of tokenized training data

        Returns:
        None
        """

        self.text_preprocessor = TextPreProcessor(training_data_file,
                                                  word2int_map_file,
                                                  sequence_length)
        self.trained_model_file = trained_model_file

    def train(self, embedding_dim=50, lstm_mem_cells=100, dense_neurons=100,
              epochs=100, batch_size=128):
        """
        Train the neural network.

        Arguments:
        self
        embedding_dim -- Embedding dimensions
        lstm_mem_cells -- Number of LSTM memory cells
        dense_neurons -- Number of neurons in the fully-connected layer
        epochs -- Number of epochs to train
        batch_size -- Batch size

        Returns:
        """

        X, Y = self.text_preprocessor.text2sequence()
        model = self._create_model(embedding_dim=embedding_dim,
                                   lstm_mem_cells=lstm_mem_cells,
                                   dense_neurons=dense_neurons)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y, epochs=epochs, batch_size=batch_size)

        # Save the trained model to file
        model.save(self.trained_model_file)

    def generate_text(self, seed_text, num_words):
        """
        Generate new text from the trained model.

        Arguments:
        self
        seed_text -- Text to seed the sequence generator
        num_words -- Number of words to generate

        Returns:
        Generated text containing num_words
        """

        model = load_model(self.trained_model_file)
        tokenizer = load(open(self.text_preprocessor.word2int_map_file, 'rb'))
        return self._generate_sequence(model, tokenizer, seed_text, num_words)

    def _generate_sequence(self, model, tokenizer, seed_text, num_words):
        """
        Generate new sequence from the trained model.

        Arguments:
        self
        model -- Trained model
        tokenizer -- word2int mapping
        seed_text -- Text to seed the sequence generator
        num_words -- Number of words to generate

        Returns:
        Generated sequence containing num_words
        """

        output_sequence = list()
        input_sequence = seed_text

        for _ in range(num_words):
            encoded_sequence = tokenizer.texts_to_sequences([input_sequence])[0]
            encoded_sequence = pad_sequences([encoded_sequence],
                                             maxlen=self.text_preprocessor.sequence_length-1,
                                             truncating='pre')
            next_word_index = model.predict_classes(encoded_sequence,
                                                    verbose=0)

            output_word = self._next_output_word(next_word_index, tokenizer)
            input_sequence += ' ' + output_word
            output_sequence.append(output_word)

        return ' '.join(output_sequence)

    def _next_output_word(self, next_word_index, tokenizer):
        """
        Generate the next output word.

        Arguments:
        self
        next_word_index -- word index in the word2int dictionary
        tokenizer -- word2int mapping

        Returns:
        Next output word
        """

        output_word = ''

        for word, index in tokenizer.word_index.items():
            if index == next_word_index:
                output_word = word
                break

        return output_word

    def _create_model(self, embedding_dim=50, lstm_mem_cells=100,
                      dense_neurons=100):
        """
        Create a Keras training model.

        Arguments:
        self
        embedding_dim -- Embedding dimensions
        lstm_mem_cells -- Number of LSTM memory cells
        dense_neurons -- Number of neurons in the fully-connected layer

        Returns:
        model -- Keras training model
        """

        model = Sequential()
        model.add(Embedding(self.text_preprocessor.vocab_size, embedding_dim,
                            input_length=self.text_preprocessor.sequence_length))
        model.add(LSTM(lstm_mem_cells, return_sequences=True))
        model.add(LSTM(lstm_mem_cells))
        model.add(Dense(dense_neurons, activation='relu'))
        model.add(Dense(self.text_preprocessor.vocab_size, activation='softmax'))

        return model

def main():
    """
    Main thread of execution.

    Arguments:
    None

    Returns:
    None
    """

    args = parse_args()
    text_generator = TextGenerator(args.training_data, args.model, args.map)

    if args.mode == 'training':
        text_generator.train(embedding_dim=50, lstm_mem_cells=100,
                             dense_neurons=100, epochs=args.epochs,
                             batch_size=args.batch)
    else:
        seed_text = TextPreProcessor(args.seed_text,
                                     args.map)._load_training_data()
        print(seed_text)
        print(text_generator.generate_text(seed_text, args.words))

def parse_args():
    """
    Parse the command line arguments.

    Arguments:
    None

    Returns:
    Parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        choices=['training', 'inference'],
                        help='Deep Learning mode')
    parser.add_argument('--training_data', required=True,
                        help='Training data file')
    parser.add_argument('--model', required=True,
                        help='Trained model file')
    parser.add_argument('--map', required=True,
                        help='word2int map file')
    parser.add_argument('--seed_text', required=True,
                        help='Seed text file for inference')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--words', type=int, default=50,
                        help='Number of generated words')

    return parser.parse_args()

if __name__ == '__main__':
    main()
