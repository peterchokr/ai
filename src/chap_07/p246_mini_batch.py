import numpy as np
import tensorflow as tf

# 데이터를 학습 데이터와 테스트 데이터로 나눈다. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

data_size = x_train.shape[0]   #train data들의 갯수 확인
print(x_train.shape)
print(data_size)
batch_size = 12	# 배치 크기

print("---")
selected = np.random.choice(data_size, batch_size)
print(selected)

# 임의로 뽑은 12개로 train data 미니 배치 구성
x_batch = x_train[selected]
y_batch = y_train[selected]
# print(x_batch, y_batch)