""" Load and process images. """

import os
import numpy as np
from Image import Image

class ImageData(object):
    def __init__(self, image_dir):
        """
        Initialize ImageData object.

        Arguments:
        self
        image_dir -- image directory

        Returns:
        None
        """

        self.image_dir = image_dir

    def load_dataset(self):
        """
        Extract and return faces from images in directory.

        Arguments:
        self

        Returns:
        List of extracted faces with labels
        """

        X, y = list(), list()
        for subdir in os.listdir(self.image_dir):
            path = self.image_dir+'/'+subdir+'/'
            if not os.path.isdir(path):
                continue
            faces = self._load_faces_from_dir(path)
            labels = [subdir for _ in range(len(faces))]
            print('>loaded %d images from class %s' % (len(faces), subdir))
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)

    def _load_faces_from_dir(self, directory):
        """
        Extract and return faces from images in directory.

        Arguments:
        self
        directory -- directory with images

        Returns:
        List of extracted faces
        """

        faces = list()
        for filename in os.listdir(directory):
            image = Image(directory+filename)
            faces.append(image.extract_face())
        return faces
