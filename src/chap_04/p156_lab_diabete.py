import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

#당뇨병 데이터셋을 적재
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

#데이터로 BMI값(인덱스=2)만 뽑아낸다.(2차원 배열로 만든다)
diabetes_X_new = diabetes_X[:, np.newaxis, 2]

#훈련 데이터와 테스트 데이터로 분리(90%:10%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new, diabetes_y, test_size=0.1, random_state=0)

#선형회귀모델로 학습한다.
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_train, y_train))

#테스트 데이터로 예측해본다
y_pred = regr.predict(X_test)

#실제 데이터와 예측 데이터를 비교
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.show


