import os
import cv2
import numpy as np

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=1)
    inverted_image = 255 - morphed_image

    contours, _ = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = cv2.drawContours(np.zeros(inverted_image.shape, np.uint8), contours, -1, 255, 1)

    cv2.imwrite(output_path, 255 - final_image)

def main():
    input_folder = './images/'
    output_folder = './result/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [
        os.path.join(input_folder, file)
        for file in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file))
    ]

    for image_path in images:
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        process_image(image_path, output_path)

if __name__ == "__main__":
    main()
