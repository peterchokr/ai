# 1. 단어별로 토큰화할 때는 word_tokenize()를 사용한다.
import nltk
from nltk.tokenize import word_tokenize

text = "This is a dog."		
print(word_tokenize(text))	


# 2. 문장 단위로 토큰화할 때는 sent_tokenize()를 사용한다.
from nltk.tokenize import sent_tokenize	# ①

text = "This is a house. This is a dog."		
print(sent_tokenize(text))	
