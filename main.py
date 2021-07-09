###### Import Libraries ######
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

import spacy
nlp = spacy.load("en_core_web_md")

print("Imports complete")

####### Data Cleaning #######
def is_not_br(token):
    return ((not ('<br' in token.text)) & (not ('/><br' in token.text)) & (not ('/>' in token.text)) & (token.text != 'br') & (not ('<' in token.text)))
def tokenize(text):
    clean_tokens = []
    for token in nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct) & is_not_br(token) & (not token.text.isdigit()):
            clean_tokens.append(token.lemma_.lower())
    return clean_tokens
# Maps 'positive' to 1 and 'negative' to 0
def convert_y(sentiments):
    converted_list = []
    for sentiment in sentiments:
        converted_list.append(1 if sentiment == 'positive' else 0)
    return np.array(converted_list)

imdb = pd.read_csv('../imdbDataset.csv')
imdb = imdb.iloc[:1500]
pd.set_option("display.max_colwidth", None)
print(f"Length : {len(imdb)}")

print("Data Cleaning Complete")

##### Word -> Number Encoding ######
##### Bag Of Words
##### Hyperparams: max_features
def bag_of_words(X, y):
    print("Starting creation of Bag of Words Model")
    bow_transformer = CountVectorizer(analyzer=tokenize, max_features=2000).fit(X)
    X = bow_transformer.transform(X)
    y = convert_y(y)
    print("Bag of Words Model Created")
    return X, y

##### TF-IDF Vectorization
##### Hyperparams: max_features
def tf_idf(X, y):
    print("Starting creation of TF-IDF Model")
    tfidf_transformer = TfidfVectorizer(analyzer=tokenize, max_features=2000).fit(X)
    X = tfidf_transformer.transform(X)
    y = convert_y(y)
    return X, y

##### Pre-trained Word Embeddings (Word2Vec)



##### BERT Language Model

##### Model Types ######
##### Logistic Regression
##### Hyperparams: train_test_split
def logistic_regression(X, y):
    print("Starting creation of Logistic Regression Model")
    logistic_model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    print("Logistic Regression Model created")
    return y_test, y_pred





##### Applying Models And Printing Accuracy #####
X, y = bag_of_words(imdb['review'], imdb['sentiment'])
y_test, y_pred = logistic_regression(X, y)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy: ", round(accuracy, 2), ", Precision: ", round(precision, 2), ", Recall ", round(recall, 2))