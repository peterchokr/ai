#AND 동작을 하는 perceptron (순수 파이썬 사용)

epsilon = 0.0000001

def perceptron(x1, x2):
    w1, w2, b = 1.0, 1.0, -1.5
    sum = x1*w1 + x2*w2 + b
    if sum > epsilon:
        return 1
    else :
        return 0
    
print(perceptron(0, 0))
print(perceptron(0, 1))
print(perceptron(1, 0))
print(perceptron(1, 1))


# 도전문제(p170). OR, NAND에 대해서도 검증해보자.