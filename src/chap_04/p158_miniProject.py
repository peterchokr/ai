import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

# 실제 아파트 가격이 아닌 가상적인 가격 
X = [[10], [18], [25], [33], [45]]		    # 단위 평
y = [3, 5, 7, 10, 15]	                   # 가격 억원

reg.fit(X, y)			# 학습

# 학습 데이터와 y 값을 산포도로 그린다. 
plt.scatter(X, y, color='black')

# 학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(X)

# 학습 데이터와 예측값으로 선그래프로 그린다. 
# 계산된 기울기와 y 절편을 가지는 직선이 그려진다. 
plt.plot(X, y_pred, color='blue', linewidth=3)		
plt.show()

area = int(input("아파트 면적을 입력하시오: "))
print(reg.predict([[area]]))