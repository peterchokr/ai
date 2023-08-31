from tensorflow.keras.preprocessing import image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img

import tensorflow as tf
my_model = tf.keras.models.load_model("MobileNet_Transfer")

img_path = 'data//' + 'd2.jpg'
myimg = image.load_img(img_path, target_size=(128,128))	# 영상 크기를 변경하고 적재한다.
x = image.img_to_array(myimg)	# 영상을 넘파이 배열로 변환한다. 
x = np.expand_dims(x, axis=0)	# 차원을 하나 늘인다. 배치 크기가 필요하다. 

image = img.imread(img_path)
plt.imshow(image)
plt.show()


# BUG !!! 
# 제대로 개, 고양이 분류가 되지 않는다.  학습모델 문제 ???
preds = my_model.predict(x)   
print("예측결과(softmax) : ", preds[0])
# print('결정 : ', np.argmax(preds))
print("예측값 최종=", preds[0].argmax())
