import numpy
import cv2


def contrast_stretching(image, coordinates):
    if image is None:
        print("The image isn't loading.")
        print("Сликата не се вчитува.")
        return None
    channels = cv2.split(image)
    mappings = [calculate_mapping(coordinates) for _ in range(3)]
    if any(mapping is None for mapping in mappings):
        return None
    transformed_channels = [apply_mapping(ch, mapping) for ch, mapping in zip(channels, mappings)]
    return cv2.merge(transformed_channels)

def calculate_mapping(coordinates):
    x = (numpy.array([point[0] for point in coordinates]))
    y = (numpy.array([point[1] for point in coordinates]))
    return numpy.polyfit(x, y, len(coordinates) - 1)


def apply_mapping(channel, mapping):
    if mapping is None:
        return channel
    mapped_channel = numpy.polyval(mapping, channel)
    clipped_channel = numpy.clip(mapped_channel, 0, 255)
    uint8_channel = clipped_channel.astype(numpy.uint8)
    return uint8_channel


print("Press 'q' to close windows.")
print("Притиснете q за да ги исклучите прозорците.")

image = cv2.imread('input_image.jpg')


coordinates = []
num = int(input("Number of coordinates:"))
for i in range(num):
    x = int(input())
    y = int(input())
    coordinates.append((x, y))

rez = contrast_stretching(image, coordinates)

cv2.imshow('original', image)
cv2.imshow('stretched', rez)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

'''
6
0
0
20
30
50
70
80
90
100
160
255
255
'''

'''
8
0
0
40
50
60
60
70
80
140
160
200
200
220
240
255
255
'''