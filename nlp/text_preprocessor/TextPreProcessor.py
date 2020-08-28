""" Pre-process text files for Deep Learning training and inference. """

from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

class TextPreProcessor:
    def __init__(self, training_data_file, word2int_map_file,
                 sequence_length=51):
        """
        Initialize text pre-processor object.

        Arguments:
        self
        training_data_file -- text file containing training data
        word2int_map_file -- file containing words to integers mapping
        sequence_length -- length of sequence of tokenized training data

        Returns:
        None
        """

        self.training_data_file = training_data_file
        self.word2int_map_file = word2int_map_file
        self.sequence_length = sequence_length
        self.vocab_size = None

    def text2sequence(self):
        """
        Convert text training data to sequences.

        Arguments:
        self

        Returns:
        X -- input sequence to the neural network
        Y -- output word
        """

        training_data = self._load_training_data()
        tokens = self._tokenize_training_data(training_data)
        sequences = self._create_sequences(tokens)
        encoded_sequences = self._encode_sequences(sequences)
        return self._create_training_set(encoded_sequences)

    def _load_training_data(self):
        """
        Load training data to memory.

        Arguments:
        self

        Returns:
        training_data -- training data
        """

        with open(self.training_data_file, 'r') as file:
            training_data = file.read()
        return training_data

    def _tokenize_training_data(self, training_data):
        """
        Tokenize training data.

        Arguments:
        self

        Returns:
        tokens -- tokenized data
        """

        tokens = text_to_word_sequence(training_data)

        # Remove tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]

        return tokens

    def _create_sequences(self, tokens):
        """
        Create sequences of length self.sequence_length from the tokens.

        Arguments:
        self
        tokens -- tokenized training data

        Returns:
        sequences -- list of sequences of length self.sequence_length each
        """

        sequences = list()
        for i in range(self.sequence_length, len(tokens)):
            sequence = tokens[i-self.sequence_length:i]
            line = ' '.join(sequence)
            sequences.append(line)

        return sequences

    def _encode_sequences(self, sequences):
        """
        Encode training sequences.

        Arguments:
        self
        sequences -- list of sequences of length self.sequence_length each

        Returns:
        encoded_sequences -- Encoded training sequences
        """

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sequences)
        encoded_sequences = tokenizer.texts_to_sequences(sequences)
        self.vocab_size = len(tokenizer.word_index) + 1

        # Save the word2int mapping to file
        dump(tokenizer, open(self.word2int_map_file, 'wb'))

        return encoded_sequences

    def _create_training_set(self, encoded_sequences):
        """
        Create training set by splitting encoded sequences into input (X) and
        output (Y).

        Arguments:
        self
        encoded_sequences -- Encoded training sequences

        Returns:
        X -- input sequence to the neural network
        Y -- output word
        """

        encoded_sequences = array(encoded_sequences)
        X, Y = encoded_sequences[:,:-1], encoded_sequences[:,-1]
        self.sequence_length = X.shape[1]

        return X, Y
