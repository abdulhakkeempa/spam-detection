from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from preprocess import prepare_dataset
import pandas as pd
import pickle

model = Pipeline([
  ('vectorizer', CountVectorizer(max_features=5000)),
  ('nb', MultinomialNB())
])

def train_model(X,y, model):
  """
    Train the model
  """
  model.fit(X, y)
  return model

if __name__ == "__main__":
  df = pd.read_csv("./dataset/spam.csv")
  X, y = prepare_dataset(df)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = train_model(X_train, y_train, model)
  print(model.score(X_test, y_test))

  with open('./model/model.pickle', 'wb') as file:
    pickle.dump(model, file)

