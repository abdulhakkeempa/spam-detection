import numpy as np
import pandas as pd
import re

def load_data():
    """
      Load the data from the file and save it to a CSV file
    """
    # Read the file
    with open('./dataset/SMSSpamCollection.txt', 'r') as file:
        data = file.read()

    # Use regex to split the data into label and text
    dataset = re.findall(r'(ham|spam)\t(.*)', data)

    # Create a DataFrame
    df = pd.DataFrame(dataset, columns=['label', 'text'])

    # Save to CSV
    df.to_csv('./dataset/spam.csv', index=False)


if __name__== "__main__":
    print("Loading data...")
    load_data()
