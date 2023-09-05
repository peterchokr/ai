# 추가 - 학습한 모델을 이용하여 내가 쓴 숫자 인식하기 

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import numpy as np

print("Loading model...")
model = load_model('mymodel')
print("Loading complete!")

import cv2
img = cv2.imread("n3.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)

img = cv2.resize(img, (28,28))
img = img.astype("float32")
img = 255-img
img = img / 255.0

plt.imshow(img.reshape(28,28), cmap="Greys")
plt.show()

pred =  model.predict(img.reshape(-1, 28, 28))
print("인식 결과 = ", pred.argmax())

cv2.waitKey(0)
cv2.destroyAllWindows()