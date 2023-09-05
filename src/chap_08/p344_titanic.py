import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# 데이터 세트를 읽어들인다. 
train = pd.read_csv("train.csv", sep=',')
test = pd.read_csv("test.csv", sep=',')

# Sex와 Pclass만 고려하고 나머지 컬럼들을 삭제한다. 
train.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',\
        'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)

# 결손치가 있는 데이터 행은 삭제한다. 
train.dropna(inplace=True)

# 기호를 수치로 변환한다. 
for ix in train.index:
    if train.loc[ix, 'Sex']=="male":
       train.loc[ix, 'Sex']=1 
    else:
       train.loc[ix, 'Sex']=0 

# 2차원 배열을 1차원 배열로 평탄화한다. 
target = np.ravel(train.Survived) 

# 생존여부를 학습 데이터에서 삭제한다. 
train.drop(['Survived'], inplace=True, axis=1)

train = train.astype(float)     

# 케라스 모델을 생성한다. 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 케라스 모델을 컴파일한다. 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 케라스 모델을 학습시킨다. 
model.fit(train, target, epochs=30, batch_size=1, verbose=1)

train_loss, train_acc = model.evaluate(train, target)
print(train_loss, train_acc)

# 추가.
# 테스트 데이터셋으로 생존여부 예측해보자
# Sex와 Pclass만 두고 나머지 컬럼들을 삭제한다.(원하다면 다른 컬럼으로)

# test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',\
#         'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)

# # 결손치가 있는 데이터 행은 삭제한다. 
# test.dropna(inplace=True)

# # 기호를 수치로 변환한다. 
# for ix in test.index:
#     if test.loc[ix, 'Sex']=="male":
#        test.loc[ix, 'Sex']=1 
#     else:
#        test.loc[ix, 'Sex']=0 

# test = test.astype(float) 

# # 테스트셋을 input_shape=(2,) 구조에 맞도록 변경한 후 실행. df에는 reshape 안됨. numpy에서 가능
# test = np.array(test)
# pred = model.predict(test.reshape(-1, 2, 1)).flatten()

# print(pred)  # activation='sigmoid'인 결과
# print(np.round(pred))  # 정수 0,1로 변환