import pandas as pd
import matplotlib.pyplot as plt

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
  
if __name__ == "__main__":
  df = pd.read_csv('./dataset/spam.csv')
  plot_frequency_count(df, 'label')
  print(groupby_data(df, 'label'))