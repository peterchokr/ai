import matplotlib.pylab as plt
from sklearn import linear_model

# 선형 회귀 모델을 생성한다. 
reg = linear_model.LinearRegression()

# 데이터는 파이썬의 리스트로 만들어도 되고 아니면 넘파이의 배열로 만들어도 됨
X = [[0], [1], [2]]		# 입력은 반드시 2차원으로.
y = [3, 3.5, 5.5]		# y = x + 3

# 학습을 시킨다. 
reg.fit(X, y)	

print(reg.coef_)		# 직선의 기울기
print(reg.intercept_) 	# 직선의 y-절편 
print(reg.score(X, y))

print(reg.predict([[5]]))

# 학습 데이터와 y 값을 산포도로 그린다. 
plt.scatter(X, y, color='black')

# 학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(X)

# 학습 데이터와 예측값으로 선그래프로 그린다. 
# 계산된 기울기와 y 절편을 가지는 직선이 그려진다. 
plt.plot(X, y_pred, color='blue', linewidth=3)		
plt.show()
