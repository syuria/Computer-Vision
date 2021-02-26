import cv2
import numpy as np
from PIL import Image
import json
import math
from transformation.autocontrast import Autocontrast
from transformation.grid_removal import GridRemoval


class RectangleDetection():
    def __init__(self):
        self.mean_w = []
        self.mean_h = []

    def remove_duplicate_val(self, image, coordinate_dict):
        """
        To remove any duplicate bounding box
        :param image: image loaded using cv2
        :param coordinate_dict: the coordinate of the bounding boxes
        :return: image and new coordinate
        """
        number = 0
        duplicate_dict = {}
        new_coordinate_dict = {}
        coordinate_dict_duplicate = coordinate_dict
        for key, value in coordinate_dict.items():  # Step 1: get 1 coordinate to check with

            if duplicate_dict.get(key) == None:  # Step 2: If the retrieved coordinate is not exist in a list of duplicate coordinate, continue to next process. If exist, just skip because we don't have to check again it is already 'removed'.
                set_range = 10
                range_xmin = range(value["coordinates"][0] - set_range, value["coordinates"][0] + set_range)
                range_ymin = range(value["coordinates"][1] - set_range, value["coordinates"][1] + set_range)
                range_xmax = range(value["coordinates"][2] - set_range, value["coordinates"][2] + set_range)
                range_ymax = range(value["coordinates"][3] - set_range, value["coordinates"][3] + set_range)

                for dkey, dvalue in coordinate_dict_duplicate.items():  # Step 3 : Compare with mirror dict.

                    if dkey >= key and duplicate_dict.get(
                            dkey) == None:  # Step 4 : Filter the key to check 1) greater than the ori key 2) not duplicate key
                        xmin = dvalue["coordinates"][0]
                        ymin = dvalue["coordinates"][1]
                        xmax = dvalue["coordinates"][2]
                        ymax = dvalue["coordinates"][3]

                        if xmin in range_xmin and ymin in range_ymin and xmax in range_xmax and ymax in range_ymax:
                            if key == dkey:
                                number += 1
                                new_coordinate_dict[number] = {"coordinates": value["coordinates"]}
                            else:
                                duplicate_dict[dkey] = {"coordinates": dvalue["coordinates"],
                                                        "coordinates_sim": value["coordinates"],
                                                        "key": key}

        for key, value in new_coordinate_dict.items():
            xmin = value["coordinates"][0]
            ymin = value["coordinates"][1]
            xmax = value["coordinates"][2]
            ymax = value["coordinates"][3]
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (36, 255, 12), 2)

        return image, new_coordinate_dict

    def find_common_w_h(self):
        """
        To select the width and height of the bounding box
        :return: width, height
        """
        w = 0
        h = 0
        w_dict = {i: self.mean_w.count(i) for i in self.mean_w}
        w_dict = dict(sorted(w_dict.items(), key=lambda item: item[1], reverse=True))
        if len(w_dict.keys())!=0:
            w = list(w_dict.keys())[0]
        h_dict = {i: self.mean_h.count(i) for i in self.mean_h}
        h_dict = dict(sorted(h_dict.items(), key=lambda item: item[1], reverse=True))
        if len(h_dict.keys()) != 0:
            h = list(h_dict.keys())[0]
        return w, h

    def filter_image_bbox(self, coordinate_dict, w, h):
        """
        To filter bounding box with certain range of height and width
        :param image: image
        :param coordinate_dict: the saved coordinate
        :param w: width
        :param h: height
        :return: new coordinate
        """
        filter_coordinate_dict = {}
        ratio_w = math.ceil(w * (25 / 100))
        ratio_h = math.ceil(h * (25 / 100))

        range_w = range(w - ratio_w, w + ratio_w)
        range_h = range(h - ratio_h, h + ratio_h)

        number = 0
        for key, value in coordinate_dict.items():
            xmin = value["coordinates"][0]
            ymin = value["coordinates"][1]
            xmax = value["coordinates"][2]
            ymax = value["coordinates"][3]
            w = xmax - xmin
            h = ymax - ymin

            if w in range_w and h in range_h:
                number += 1
                filter_coordinate_dict[number] = {"coordinates": value["coordinates"]}

        return filter_coordinate_dict

    def rectangle_detection(self, image, image_dest):
        """
        To detect rectangle in legend
        :param image: an image loaded using openCV
        :param image_dest: the folder destination to save the output (legend.png, legend.json)
        :return: None
        """
        ori_image = image

        grid_removal = GridRemoval()
        image = grid_removal.morp_main(image)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        auto_contrast = Autocontrast()
        image = auto_contrast.preprocess_images(im_pil, 0.6, None)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        h, w, c = ori_image.shape
        if w > 1000:
            min_area = 100
        else:
            min_area = 600

        max_area = 5000

        image_number = 0
        coordinate_dict = {}

        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                image_number += 1
                coordinate_dict[image_number] = {"coordinates": [x, y, x + w, y + h]}
                self.mean_w.append(w)
                self.mean_h.append(h)

        w, h = self.find_common_w_h()
        coordinate_dict = self.filter_image_bbox(coordinate_dict, w, h)

        ori_image, coordinate_dict = self.remove_duplicate_val(ori_image, coordinate_dict)

        with open(image_dest + "legend.json", "w") as fp:
            json.dump(coordinate_dict, fp)

        cv2.imwrite(
            image_dest + "legend.png",
            ori_image,
        )