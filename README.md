# Spam Detection
This project utilizes a recent <a href="https://www.kaggle.com/datasets/indronil2489/spam-sms-dataset">spam dataset</a> from Kaggle to build a spam classifier. The text data is preprocessed through various techniques including `stop words` removal and `lemmatization`. A pipeline is created with `CountVectorizer` and `Multinomial Naive Bayes` classifier, which achieved an impressive accuracy of 98%. The trained model is then deployed as a `FastAPI` endpoint for real-time spam classification. 

## Packages
1. Sklearn
2. Pandas
3. FastAPI

## NLP Basic Steps
1. Data Preprocessing  
    1.1 Changing the characters to lowercase  
    1.2 Tokenization  
    1.2 Stemming  
     
     
## Bayes Theorem
$$
P(A|B) = \frac{P(B|A) * P(A)}{P(B)}
$$

## Spam Classification
Let $w1,w2,.....,wn$ be the words contained in the given message/email.  
The probability that the message is spam given the words can be written as:

$$
P(spam|w1 \cap w2 \cap w3 ..... \cap wn ) = \frac{P(w1 \cap w2 \cap w3 ..... \cap wn | spam) * P(spam)}{P(w1 \cap w2 \cap w3 ..... \cap wn )}
$$

If we assume the occurrences of the words are independent of the other words, the formula can be rewritten as:

$$
P(spam|w1 \cap w2 \cap w3 ..... \cap wn ) = \frac{P(w1 | spam) * P(w2 | spam) * P(w3 | spam) ....... P(wn | spam) * P(spam)}{P(w1 \cap w2 \cap w3 ..... \cap wn )}
$$


## Setup

1. Clone the project repository:
    ```
    git clone https://github.com/abdulhakkeempa/spam-detection.git
    ```

2. Navigate to the project directory:
    ```
    cd spam-detection
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Exploratory Data Analysis (EDA)

1. Run the `eda.py` script. This will generate word cloud images and save them to a folder. Make sure to create an 'images' folder in the directory prior to running the script. This script will also display a bar chart for spam and ham messages:
    ```
    python eda.py
    ```

## Training

1. Before running the training script, create a 'model' folder in the directory.
2. Run the `train.py` script. This will preprocess the data and train the Naive Bayes model:
    ```
    python train.py
    ```

## Testing

1. Run the `evaluate.py` script. This will test the model using the test part of the dataset:
    ```
    python evaluate.py
    ```

## Custom Evaluation

1. To evaluate the model with a custom input message, run the `test.py` script with the `--message` option followed by your custom message:
    ```
    python test.py --message="custom input message"
    ```

## Running the FastAPI Server

1. Run the FastAPI server using Uvicorn with the `--reload` option:
    ```
    uvicorn main:app --reload
    ```

2. You can test the API by navigating to `localhost:8000/docs` in your web browser.

3. Using CURL
```
curl -X POST "http://localhost:8000/predict" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"message\":\"your message here\"}"
```
