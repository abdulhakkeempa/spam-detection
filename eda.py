import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from math import log, sqrt

def plot_frequency_count(df, column):
  """
    Plot the frequency count of the column
  """
  if column in df.columns:
    df[column].value_counts().plot(kind='bar')
    plt.show()

  else :
    print("Column not found")

def groupby_data(df, column):
  """
    Group the data by the column
  """
  if column in df.columns:
    return df.groupby(column).count()
  else :
    return "Column not found"
  
def wordcloud(df, category):
  assert category in ['spam', 'ham'], "Category should be either spam or ham"
  spam_words = " ".join(list(df[df['label'] == category]['text']))
  spam_wc = WordCloud(width=512, height=512).generate(spam_words)
  plt.figure(figsize=(10,8), facecolor='k')
  plt.imshow(spam_wc)
  plt.axis('off')
  plt.title("Wordcloud for {}".format(category), fontsize=20)
  plt.tight_layout(pad=0)
  plt.savefig('./images/wordcloud_{}.png'.format(category))
  plt.show()


if __name__ == "__main__":
  df = pd.read_csv('./dataset/spam.csv')
  plot_frequency_count(df, 'label')
  wordcloud(df, 'spam')
  wordcloud(df, 'ham')
  print(groupby_data(df, 'label'))