from sklearn.model_selection import train_test_split
from test import load_model
from preprocess import prepare_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

model = load_model()
df = pd.read_csv("./dataset/spam.csv")
X, y = prepare_dataset(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
print(f"Training Score : {accuracy_score(model.predict(X_train), y_train)}")
print(f"Test Score : {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

