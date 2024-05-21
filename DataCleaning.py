import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("data/food_recipes.csv")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    return text
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    sentence = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return sentence
def stopwords_removal(text):
    stop_words = set(stopwords.words('english'))
    sentence = ' '.join([word for word in text.split() if word not in stop_words])
    return sentence
data['directions'] = data['directions'].apply(clean_text)
data['directions'] = data['directions'].apply(lemmatize)
data['directions'] = data['directions'].apply(stopwords_removal)

print(data['directions'])