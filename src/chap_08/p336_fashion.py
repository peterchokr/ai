import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.imshow(train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_images[0].shape)
print('정확도:', test_acc)


# 기능 추가. 사용자 이미지로 테스트 해보기
# OpenCV 설치 - pip install opencv-python

import cv2 as cv

image = cv.imread('fash_03.jpg', cv.IMREAD_GRAYSCALE)

image = cv.resize(image, (28, 28))
image = image.astype('float32')
image = 255-image
image /= 255.0

plt.imshow(image,cmap='Greys')
plt.show()

image = image.reshape(-1, 28, 28)   # 입력 레이어 (none, 28, 28)에 맞도록 shape 변경

pred = model.predict(image)
print(pred)

category = ['티셔츠', '바지', '머리부터 뒤집어 써서 입는 스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '부츠']
print("추정된 카테고리 : ", pred.argmax(), "->", category[pred.argmax()])