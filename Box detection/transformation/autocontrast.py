import cv2
from PIL import Image
import numpy as np
from PIL import ImageOps

class Autocontrast():

    def normalize(self, img, cutoff=0, ignore=None):
        """
        To run the autocontrast on the image
        :param img: a grayscale image
        :param cutoff: the percentage to remove and strengthen the black and white colour
        :param ignore: the image background consideration
        :return: an image
        """
        try:
            return ImageOps.autocontrast(img, cutoff=cutoff, ignore=ignore)
        except IOError:
            return img

    def to_grayscale(self, img):
        """
        To convert into grayscale image with L mode.
        :param img: an image loaded using PIL
        :return: a grayscale image
        """
        grayscale = img.convert('L')
        return grayscale

    def preprocess_images(self, img, cutoff, ignore):
        """
        To apply autocontrast on image
        :param img: an image loaded using
        :param cutoff:
        :param ignore:
        :return:
        """
        noisy_img = img
        result = self.to_grayscale(noisy_img)
        result = self.normalize(result, cutoff, ignore)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result