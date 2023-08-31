# 신장과 체중을 받아서 성별을 출력하는 학습(알고리즘)을 직접 코딩

import numpy as np

epsilon = 0.0000001		# 부동소수점 오차 방지

def step_func(t):		# 퍼셉트론의 활성화 함수
    if t > epsilon: return 1
    else: return 0

X = np.array([			# 훈련 데이터 세트
    [160, 55, 1],			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다. 
    [163, 43, 1],			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다. 
    [165, 48, 1],			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다. 
    [170, 80, 1],			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.     
    [175, 76, 1],			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.     
    [180, 70, 1]			# 맨 끝의 1은 바이어스를 위한 입력 신호 1이다. 
])
	
y = np.array([0, 0, 0, 1, 1, 1])	# 정답을 저장하는 넘파이 행렬
W = np.zeros(len(X[0]))	# 가중치를 저장하는 넘파이 행렬

def perceptron_fit(X, Y, epochs=10):	# 퍼셉트론 학습 알고리즘 구현
    global W
    eta = 0.2				# 학습률

    for t in range(epochs):
        print("epoch=", t, "======================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict		# 오차 계산
            W += eta * error * X[i]		# 가중치 업데이트
            print("현재 처리 입력=",X[i],"정답=",Y[i],"출력=",predict,"변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y):		# 예측	
    global W
    for x in X:
         print(x[0], x[1], "->", step_func(np.dot(x, W)))

perceptron_fit(X, y, 22)
perceptron_predict(X, y)

# 테스트를 수행
print("===")
perceptron_predict([[180,89,1]],0)