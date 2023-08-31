# 불용어 처리 
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

print(stopwords.words('english')[:20])