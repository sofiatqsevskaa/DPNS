import cv2
import numpy
import matplotlib.pyplot

def apply_compass_filters(image):
    filters = {
        'N': numpy.array([[-1, -1, -1],
                          [1, 2, 1],
                          [0, 0, 0]]),
        'NE': numpy.array([[-1, -1, 1],
                           [-1, 2, 1],
                           [0, 0, 0]]),
        'E': numpy.array([[0, -1, 1],
                          [0, 2, 1],
                          [0, -1, 1]]),
        'SE': numpy.array([[0, 0, 0],
                           [-1, 2, 1],
                           [-1, -1, 1]]),
        'S': numpy.array([[0, 0, 0],
                          [1, 2, 1],
                          [-1, -1, -1]]),
        'SW': numpy.array([[0, 0, 0],
                           [1, 2, -1],
                           [1, -1, -1]]),
        'W': numpy.array([[1, -1, 0],
                          [1, 2, 0],
                          [1, -1, 0]]),
        'NW': numpy.array([[1, -1, -1],
                           [1, 2, -1],
                           [0, 0, 0]])
    }

    outputs = {}

    for direction, filter in filters.items():
        filtered_image = cv2.filter2D(image, -1, filter)
        outputs[direction] = filtered_image

    return outputs


def display_images(images, title):
    matplotlib.pyplot.figure(figsize=(12, 6))
    for i, (key, image) in enumerate(images.items(), 1):
        matplotlib.pyplot.subplot(2, 4, i)
        matplotlib.pyplot.imshow(image, cmap='gray')
        matplotlib.pyplot.title(key)
        matplotlib.pyplot.axis('off')
    matplotlib.pyplot.suptitle(title)
    matplotlib.pyplot.show()


image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found. Check the file path.")

outputs = apply_compass_filters(image)

display_images(outputs, "Outputs of each Compass Filter")

combined_image = numpy.zeros_like(list(outputs.values())[0])
for image in outputs.values():
    combined_image = numpy.maximum(combined_image, image)

matplotlib.pyplot.figure(figsize=(12, 4))
thresholds = [50, 100, 150]
for i, thresh in enumerate(thresholds, 1):
    _, thresh_image = cv2.threshold(combined_image, thresh, 255, cv2.THRESH_BINARY)
    matplotlib.pyplot.subplot(1, 3, i)
    matplotlib.pyplot.imshow(thresh_image, cmap='gray')
    matplotlib.pyplot.title(f'Threshold = {thresh}')
    matplotlib.pyplot.axis('off')
matplotlib.pyplot.show()
