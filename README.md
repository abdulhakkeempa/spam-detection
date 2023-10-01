# Spam Detection

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

