import cv2
import numpy as np

class GridRemoval():
    def morp_main(self, img):
        """
        To remove long horizontal and vertical line
        :param img: an image loaded using openCV
        :return: an image without long horizontal and vertical line
        """
        image = self.run_morp_2v25(img)
        image = self.run_morp_2h25(image)
        return image

    def run_morp_2v25(self, img):
        """
        To remove long vertical line
        :param img: an image loaded with openCV
        :return: an image without vertical line
        """
        noisy_img = img
        cleaned = cv2.bitwise_not(noisy_img)
        k=25
        kernel_line = np.ones((k, 1), np.uint8)
        return self.run_morp(cleaned,kernel_line)

    def run_morp_2h25(self, img):
        """
        To remove long horizontal line
        :param img: an image loaded with openCV
        :return: an image without horizontal line
        """
        noisy_img = img
        cleaned = cv2.bitwise_not(noisy_img)
        k=25
        kernel_line = np.ones((1, k), np.uint8)
        return self.run_morp(cleaned, kernel_line)

    def run_morp(self, cleaned, kernel_line):
        """
        To remove the gridline with defined kernel line.
        :param cleaned: a pre processed image with inverted colour
        :param kernel_line: kernal value to define line
        :return: an image without horizontal/vertical line
        """
        clean_lines = cv2.erode(cleaned, kernel_line, iterations=6)
        clean_lines = cv2.dilate(clean_lines, kernel_line, iterations=6)
        # (3) Subtract lines:
        cleaned_img_without_lines = cleaned - clean_lines
        cleaned_img_without_lines = cv2.bitwise_not(cleaned_img_without_lines)

        return cleaned_img_without_lines