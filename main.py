from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
from deep_translator import GoogleTranslator

filename = 'images/test3.jpg'
#searchText = 'Three ear caps of'
#searchText = 'different sizes'
searchText = 'IPX7 Waterproof'
#searchText = 'USB cable'
#searchText = 'Noise Reduction'

def get_text_color_in_square(image, top_left, bottom_right):
    square = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    mean_color = cv2.mean(square)[:3]  # Возвращает средний цвет в BGR
    return mean_color

def translate_Text_On_Image(image,d,i,wordsIn):
    resultImage=image.copy()
    lenWords=len(wordsIn)
    #Получение координат словосочетания
    x = d['left'][i]
    y = d['top'][i]
    for k in range(lenWords):
        if y>d['top'][i+k]:
            y=d['top'][i+k]
    w = d['left'][i+lenWords-1]- d['left'][i] + d['width'][i+lenWords-1]
    h = d['height'][i]
    for k in range(lenWords):
        if h < d['height'][i + k]:
            h = d['height'][i + k]
    # Восстановление фона
    mask = np.zeros(resultImage.shape[:2], dtype=np.uint8)
    mask[y - 1:y + h + 1, x - 1:x + w + 1] = 255
    resultImage = cv2.inpaint(resultImage, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # Новый текст и его параметры
    translation = GoogleTranslator(source='auto', target='ru').translate(searchText)
    font_color = get_text_color_in_square(image, (x, y), (x + w, y + h)) #(B,G,R)
    font_scale=0.365 #В ручную подоброн более-менее нормальный размер текста
    (font, thickness, line_type) = (cv2.FONT_HERSHEY_COMPLEX, 1, cv2.LINE_AA)
    text_size = cv2.getTextSize(translation, font, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2  # Центрируем по ширине
    text_y = y + (h + text_size[1]) // 2  # Центрируем по высоте
    cv2.putText(resultImage, translation, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
    print(x,y,w,h)
    print(font_scale,len(searchText),len(translation))
    return resultImage



image = cv2.imread(filename)
image = cv2.resize(image, None, fx = 0.8, fy = 0.8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename2 = "images/test4.png".format(os.getpid())
cv2.imwrite(filename2, gray)

d = pytesseract.image_to_data(Image.open(filename2), output_type=pytesseract.Output.DICT)
n_boxes = len(d['text'])

wordsIn = searchText.split()

for i in range(n_boxes):
    if d['text'][i] == wordsIn[0] and i<=n_boxes - len(wordsIn):
        find=True
        for j in range(len(wordsIn)):
            if d['text'][i+j]!=wordsIn[j]:
                find=False
                break
        if find:
            image=translate_Text_On_Image(image,d,i,wordsIn)

os.remove(filename2)
cv2.imshow("Output", image)
cv2.waitKey(0)


