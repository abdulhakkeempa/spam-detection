from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re


def preprocess_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
  """
    Preprocess the message
  """

  message = re.sub(r'\\d+', '', message) #remove numbers

  if lower_case:
    message = message.lower()

  words = word_tokenize(message)
  words = [w for w in words if len(w) > 2] #words with minimum length 3

  if gram>2:
    words = []
    for i in range(len(words) - gram + 1):
      words += [' '.join(words[i:i+gram])]
    
  if stop_words:
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]

  if stem:
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

  joined_words = ' '.join(words)
  return joined_words


def prepare_dataset(df):
  """
    Prepare the dataset
  """
  X = df['text'].apply(preprocess_message)
  y = df['label'].replace({'ham':0, 'spam':1}) #ham = 0, spam = 1
  return X, y

if __name__ == "__main__":
  import nltk
  nltk.download('punkt', download_dir = "E:\\spam-classification\\venv\\nltk_data")
  nltk.download('stopwords', download_dir = "E:\\spam-classification\\venv\\nltk_data")

  df = pd.read_csv("./dataset/spam.csv")
