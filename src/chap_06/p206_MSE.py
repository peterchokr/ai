# 평균제곱오차(Mean Square Error)를 계산하는 예
# 정답 target에 작은 오차를 가지는 y1과 큰 오차를 가지는 y2
import numpy as np

def MSE(target, y):
    return 0.5 * np.sum((y-target)**2)

y1 = np.array([0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
y2 = np.array([0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

target = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,])

print(np.round(MSE(target, y1),2))
print(np.round(MSE(target, y2),2))