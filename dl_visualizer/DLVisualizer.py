""" Visualize Deep Learning elements such as model, accuracy, loss etc. """

import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

class DLVisualizer:
    def __init__(self, model, history):
        """
        Initialize DLVisualizer object.

        Arguments:
        self
        model -- Keras model
        history -- record of training loss values and metrics values at successive epochs

        Returns:
        None
        """

        self.model = model
        self.history = history

    def visualize_model(self, model_plot_file):
        """
        Visualize the Keras model.

        Arguments:
        self
        model_plot_file -- file where the model plot is saved

        Returns:
        None
        """

        plot_model(self.model, to_file=model_plot_file, show_shapes=True, show_layer_names=True)

    def visualize_model_accuracy(self, model_accuracy_file):
        """
        Visualize model accuracy.

        Arguments:
        self
        model_accuracy_file -- file where the accuracy plot is saved

        Returns:
        None
        """

        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.ylim(0, 1)
        plt.savefig(model_accuracy_file)
        plt.clf()

    def visualize_model_loss(self, model_loss_file):
        """
        Visualize model loss.

        Arguments:
        self
        model_loss_file -- file where the loss plot is saved

        Returns:
        None
        """

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.ylim(0, 1)
        plt.savefig(model_loss_file)
        plt.clf()
