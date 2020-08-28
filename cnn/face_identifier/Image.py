""" Deep Learning system for extracting face from image. """

import PIL
import numpy as np
from mtcnn.mtcnn import MTCNN

class Image(object):
    def __init__(self, image):
        """
        Initialize Image object.

        Arguments:
        self
        image -- image to be processed

        Returns:
        None
        """

        self.image = image

    def extract_face(self):
        """
        Extract face from image.

        Arguments:
        self

        Returns:
        Resized face NumPy array
        """

        pixels = self._preprocess_image()
        x1, y1, x2, y2 = self._detect_face(pixels)
        return self._extract_face_array(pixels, x1, y1, x2, y2)

    def _preprocess_image(self):
        """
        1. Convert to RGB
        2. Convert image to NumPy array

        Arguments:
        self

        Returns:
        NumPy array representation of image
        """

        rgb_image = PIL.Image.open(self.image).convert('RGB')
        return np.asarray(rgb_image)

    def _detect_face(self, pixels):
        """
        Detect face in image.

        Arguments:
        self
        pixels -- NumPy array representation of image

        Returns:
        Co-ordinates of face
        """

        mtcnn = MTCNN()
        faces = mtcnn.detect_faces(pixels)
        x1, y1, width, height = faces[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1+width, y1+height

        return x1, y1, x2, y2

    def _extract_face_array(self, pixels, x1, y1, x2, y2):
        """
        Extract face in image.

        Arguments:
        self
        pixels -- NumPy array representation of image
        x1, y1, x2, y2 -- Co-ordinates of face

        Returns:
        Resized face array
        """

        face = pixels[y1:y2, x1:x2]
        resized_face = PIL.Image.fromarray(face).resize((160, 160))
        return np.asarray(resized_face)
