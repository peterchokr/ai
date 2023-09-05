import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 사전 훈련된 모델 중에서 MobileNet을 생성한다
# imagenet으로 학습된 가중치 다운로드를 가져오고(weights='imagenet'), 분류기 레이어는 생성하지 않음(include_top=False).
base_model=MobileNet(weights='imagenet', include_top=False) 

# MobileNet의 출력을 추가 레이어에 연결한다
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(1024,activation='relu')(x) 
x=Dense(512,activation='relu')(x) 
preds=Dense(2,activation='softmax')(x) 

model=Model(inputs=base_model.input, outputs=preds)

# 모델이 가진 레이어 중에서 20번째까지는 변경되지 않도록 설정
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

# 개와 고양이 이미지를 이용하여 데이터 증대
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator=train_datagen.flow_from_directory('./Petimages/', 
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
# 개와 고양이를 분류하도록 학습하고 모델 저장
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)

model.save("MobileNet_Transfer")
