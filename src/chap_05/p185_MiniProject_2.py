# 신장과 체중을 받아서 성별을 출력하는 학습(알고리즘)을 sklearn을 사용하여 코딩

from sklearn.linear_model import Perceptron

# 훈련 데이터 세트와 정답
X = [[160, 55], [163, 43], [165, 48], [170, 80], [175, 76], [180, 70]]
y = [0, 0, 0, 1, 1, 1]

# 퍼셉트론을 생성. tol은 종료조건, random_state는 난수의 시드
clf = Perceptron(tol=1e-3, random_state=0)

# 학습을 수행
clf.fit(X, y)

# 테스트를 수행
print(clf.predict([[180, 89]]))


