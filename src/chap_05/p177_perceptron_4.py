# 가중치 W를 구하는 학습(알고리즘)을 sklearn을 사용하여 코딩

from sklearn.linear_model import Perceptron

# 논리적 AND 연산 샘플과 정답
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 1]

# 퍼셉트론을 생성. tol은 종료조건, random_state는 난수의 시드
clf = Perceptron(tol=1e-3, random_state=0)

# 학습을 수행
clf.fit(X, y)

# 테스트를 수행
print(clf.predict(X))

# 분석 결과 평가 실시
import sklearn.metrics as mt
score = mt.accuracy_score(y, clf.predict(X))
print(score)