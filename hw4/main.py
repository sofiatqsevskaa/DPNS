import os
import cv2
import numpy as np

def process_image(img_path):
    # Читање на сликата
    image = cv2.imread(img_path)
    # Претворање на сликата во сива скала
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Замаглување на сликата
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
    # Применување на бинарен праг со OTSU метод
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфолошко отворање на сликата
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
    # Инвертирање на сликата
    inverted_image = 255 - morphed_image
    # Наоѓање на контури
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Цртање на контурите на празна слика
    final_image = cv2.drawContours(np.zeros(inverted_image.shape, np.uint8), contours, -1, 255, 1)
    # Создавање на директориумот 'results' ако не постои
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    # Запишување на конечната слика во директориумот 'results'
    cv2.imwrite(os.path.join(results_dir, os.path.basename(img_path)), final_image)
    return contours[0]

def calculate_similarities(query_image, input_folder):
    # Сличности е празен речник за чување на сличностите
    similarities = {}
    # Список на сите слики во директориумот 'images'
    images = [os.path.join(input_folder, i) for i in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, i))]
    # Пресметување на сличностите за секоја слика
    for image in images:
        contours = process_image(image)
        similarity = cv2.matchShapes(query_image, contours, cv2.CONTOURS_MATCH_I1, 0)
        similarities[os.path.basename(image)] = similarity
    return similarities

def main():
    # Патеките до директориумите
    query_folder = './query/'
    input_folder = './images/'
    # Кориснички внес за име на сликата за пребарување
    query_image_name = input('Image: ')
    query_image_path = os.path.join(query_folder, query_image_name)
    # Проверка дали сликата за пребарување постои
    if not os.path.isfile(query_image_path):
        print(f"Query image '{query_image_name}' not found in '{query_folder}'")
        return
    # Обработка на сликата за пребарување
    query_image = process_image(query_image_path)
    # Пресметување на сличностите
    similarities = calculate_similarities(query_image, input_folder)
    # Печатење на сличностите сортирани по вредност
    for img_name, similarity in sorted(similarities.items(), key=lambda x: x[1]):
        print(f'{img_name}:\t{similarity}')

if __name__ == '__main__':
    main()
