import cv2
import argparse
from RectangleDetection import RectangleDetection

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", type=str, default="1.png", help="path to legend image")
ap.add_argument("-o", "--image_dest", type=str, default="output_image/", help="path to output image")
args = vars(ap.parse_args())

image_path = args["image_path"]
image_dest = args["image_dest"]
image = cv2.imread(image_path)
rd = RectangleDetection()
rd.rectangle_detection(image, image_dest)