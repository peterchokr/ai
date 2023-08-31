import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text_data="""Soft as the voice of an angel\n
Breathing a lesson unhead\n
Hope with a gentle persuasion\n
Whispers her comforting word\n
Wait till the darkness is over\n
Wait till the tempest is done\n
Hope for sunshine tomorrow\n
After the shower
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])   #단어사전 만들기
encoded = tokenizer.texts_to_sequences([text_data])[0]   #정수 인코딩
print(encoded)

print(tokenizer.word_index)    #단어사전 출력
vocab_size = len(tokenizer.word_index) + 1
print('어휘 크기: %d' % vocab_size)

# 훈련용 단어 시퀀스 만들기
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print(sequences)
print('총 시퀀스 개수: %d' % len(sequences))

# 훈련 데이터와 정답 만들기
sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]
print("X=", X)
print("y=", y)

# 학습모델 구성 및 학습하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))  # 임베딩 단어벡터 크기 = 10
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=2)

# 테스트 단어를 정수 인코딩한다. 
test_text = 'Wait'
encoded = tokenizer.texts_to_sequences([test_text])[0]
encoded = np.array(encoded)
print(encoded)

# 신경망의 예측값을 출력한다. 
onehot_output = model.predict(encoded)
print('onehot_output=', onehot_output)

# 가장 높은 출력을 내는 유닛을 찾는다. 
output = np.argmax(onehot_output)
print('output=', output)

# 단어사전으로 출력층의 유닛 번호를 단어로 바꾼다. 
# print(tokenizer.word_index)
print(test_text, "=>", end=" ")
for word, index in tokenizer.word_index.items():
	if index == output:
		print(word)