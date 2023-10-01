import pickle
from preprocess import preprocess_message
import argparse

def load_model():
    """
      Load the model from the file
    """
    with open('./model/model.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

CLASSES = ['ham', 'spam']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the label of the message')
    parser.add_argument('--message', help='Enter the message to predict the label', required=True)
    args = parser.parse_args()
    model = load_model()
    message = args.message
    message = preprocess_message(message)
    X = []
    X.append(message)
    print(model.predict(X))
    print(f"The predicted class is {CLASSES[model.predict(X)[0]]}")
    print(model.predict_proba(X))