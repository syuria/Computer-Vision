import numpy as np
import cv2
from PIL import ImageOps, Image

def run_morph2(cleaned, new_name):
    # vertical kernel:
    kernel_line = np.ones((25, 1), np.uint8)
    clean_lines_v = cv2.erode(cleaned, kernel_line, iterations=6)
    clean_lines_v = cv2.dilate(clean_lines_v, kernel_line, iterations=6)
    # (3) Subtract lines:
    cleaned_img_without_lines = cleaned - clean_lines_v
    cleaned_img_without_lines = cv2.bitwise_not(cleaned_img_without_lines)
    cv2.imwrite(new_name, cleaned_img_without_lines)

def normalize(img, cutoff=0):
    try:
        return ImageOps.autocontrast(img, cutoff=cutoff)
    except IOError:
        return img

def to_grayscale(img):
    grayscale = img.convert('L')
    return grayscale

images = ['1.png',
          '4.png',
          '10.png']

for img in images:
    noisy_img = cv2.imread(img)
    cleaned = cv2.bitwise_not(noisy_img)
    image_name = img.rsplit('.')[0]
    new_name = image_name + '_2v25.png'
    run_morph2(cleaned, new_name)

    denoised_img = new_name
    noisy_img = Image.open(denoised_img)
    result = to_grayscale(noisy_img)
    result = normalize(result)
    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    image_name = denoised_img.rsplit('.')[0]
    new_name = image_name + '-autocontrast.png'
    cv2.imwrite(new_name, result)
