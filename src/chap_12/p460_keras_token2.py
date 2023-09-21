from tensorflow.keras.preprocessing.text import Tokenizer

t  = Tokenizer()
text = """Deep learning is part of a broader family of machine learning methods 
	based on artificial neural networks with representation learning. """

t.fit_on_texts([text])   # 단어사전 만들기
print("단어집합 : ", t.word_index)    # 단어사전 출력하기

# 텍스트를 인덱스 시퀀스(정수 인코딩)로 변환하기
seq = t.texts_to_sequences([text])[0]   
print(text,"->", seq) 