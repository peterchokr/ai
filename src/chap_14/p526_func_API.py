# MNIST 숫자 이미지를 처리하는 신경망(함수형 API로 작성)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 함수형 API로 모델 정의
inputs = keras.Input(shape=(784,))			# (1)
dense = layers.Dense(64, activation="relu")		# (2)
x = dense(inputs)					# (3)
x = layers.Dense(64, activation="relu")(x)		# (4)
outputs = layers.Dense(10)(x)				# (5)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255   # 60000개의 훈련 데이터를 1차원 벡터(784)로 평탄화
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)  # 학습

test_scores = model.evaluate(x_test, y_test, verbose=2)  # 평가
print(test_scores)

